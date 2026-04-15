"""Load WMT MQM sentence-level scores from HuggingFace.

Dataset: RicardoRei/wmt-mqm-human-evaluation
Schema: lp, src, mt, ref, score, system, annotators, domain, year

The `score` column is an aggregated MQM penalty (negative; 0 = perfect,
more negative = more errors). We convert to quality_score in [0, 1] using
per-language min-max normalization so it is compatible with the unified
quality_score column used throughout the pipeline.

NOTE: This dataset provides sentence-level scores only. Span-level category
annotations (accuracy vs. fluency breakdown) are not available from this
source. The category correlation analysis is disabled for this tier.

Available pairs (confirmed): en-de (WMT20-23), zh-en (WMT20-23),
                              en-ru (WMT21/22), he-en (WMT23), en-es (WMT24).
"""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from mqmbench.constants import AnnotationTier
from mqmbench.data.utils import build_segment_id, lang_from_pair, rename_wmt_columns

logger = logging.getLogger(__name__)


def _normalize_per_lang(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize MQM scores to [0, 1] within each language."""
    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

    df = df.copy()
    df["quality_score"] = df.groupby("lang")["mqm_score"].transform(_minmax)
    return df


def load_wmt_mqm(lang_pairs: list[str] | None = None) -> pd.DataFrame:
    """Load Tier 1a sentence-level MQM scores.

    Args:
        lang_pairs: Language pairs to load (e.g. ['en-de', 'zh-en']).
                    If None, loads all available pairs.

    Returns:
        DataFrame with one row per segment, columns:
            segment_id, source, hypothesis, reference, lang,
            mqm_score, quality_score, annotation_tier.
        Does NOT pass through converter.py (sentence-level scores provided directly).
    """
    logger.info("Loading WMT MQM dataset from HuggingFace...")
    df = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train").to_pandas()

    available = sorted(df["lp"].unique().tolist())
    logger.info(f"WMT MQM available pairs: {available}")

    if lang_pairs:
        missing = [lp for lp in lang_pairs if lp not in available]
        if missing:
            logger.warning(f"Requested pairs not found in dataset: {missing}")
        df = df[df["lp"].isin(lang_pairs)].copy()

    if df.empty:
        logger.warning("No MQM data matched the requested language pairs.")
        return pd.DataFrame(columns=[
            "segment_id", "source", "hypothesis", "reference",
            "lang", "mqm_score", "quality_score", "annotation_tier",
        ])

    df = rename_wmt_columns(df)
    df = df.rename(columns={"score": "mqm_score"})
    df["segment_id"] = build_segment_id(df)
    df["lang"] = df["lp"].apply(lang_from_pair)
    df = _normalize_per_lang(df)
    df["annotation_tier"] = AnnotationTier.HUMAN_MQM

    # Keep year + system for the medium>high anomaly analysis (multi-year variance).
    keep = ["segment_id", "source", "hypothesis", "reference",
            "lang", "mqm_score", "quality_score", "annotation_tier",
            "domain", "year", "system"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Span-level loader (google/wmt-mqm-human-evaluation TSV files)
# ---------------------------------------------------------------------------
# TSV schema (columns vary slightly by year but core fields are consistent):
#   system, doc, doc_id, seg_id, rater, source, target, category, severity
#
# category values look like "Accuracy/Mistranslation", "Fluency/Grammar", etc.
# severity values: "major", "minor", "neutral" (neutral rows are no-error placeholders)
#
# Download the TSVs from https://github.com/google/wmt-mqm-human-evaluation
# and point data.wmt_mqm_span_dir in settings.toml at the local directory.

_LANG_FROM_FILENAME = {
    "ende": "de", "zhen": "zh", "enru": "ru", "heen": "he", "enes": "es",
    "deen": "de", "ruen": "ru", "enzh": "zh",
}


def load_wmt_mqm_spans(
    span_dir: str | Path,
    lang_pairs: list[str] | None = None,
) -> pd.DataFrame:
    """Load raw WMT MQM span annotations from Google's TSV release.

    Args:
        span_dir: Directory containing the downloaded TSV files
                  (e.g. data/wmt_mqm_spans/).
        lang_pairs: Language pairs to keep (e.g. ['en-de', 'zh-en']).
                    If None, loads all found files.

    Returns:
        DataFrame with span-level annotation rows compatible with converter.py:
            segment_id, source, hypothesis, reference, lang,
            error_type, severity, error_start, error_end, annotation_tier.
    """
    span_dir = Path(span_dir)
    tsv_files = sorted(span_dir.glob("**/*.tsv")) + sorted(span_dir.glob("**/*.csv"))
    if not tsv_files:
        logger.warning(f"No TSV files found in {span_dir}")
        return pd.DataFrame()

    # Build set of target langs from requested pairs
    target_langs: set[str] | None = None
    if lang_pairs:
        target_langs = {lang_from_pair(lp) for lp in lang_pairs}

    frames = []
    for f in tsv_files:
        try:
            raw = pd.read_csv(f, sep="\t", dtype=str, on_bad_lines="skip").fillna("")
        except Exception as exc:
            logger.warning(f"Could not read {f}: {exc}")
            continue

        # Normalise column names across WMT years
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
        col_map = {
            "target": "hypothesis", "mt": "hypothesis", "hyp": "hypothesis",
            "src": "source", "ref": "reference",
            "category": "error_type", "severity": "severity",
            "seg_id": "seg_id",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        # Infer lang from filename if not a column
        if "lang" not in raw.columns and "lp" not in raw.columns:
            stem = f.stem.lower().replace("-", "").replace("_", "")
            lang = next((v for k, v in _LANG_FROM_FILENAME.items() if k in stem), None)
            if lang is None:
                logger.warning(f"Could not infer lang from filename {f.name}, skipping")
                continue
            raw["lang"] = lang
        elif "lp" in raw.columns:
            raw["lang"] = raw["lp"].apply(lang_from_pair)

        if target_langs and not raw["lang"].isin(target_langs).any():
            continue

        # Filter to requested langs
        if target_langs:
            raw = raw[raw["lang"].isin(target_langs)].copy()

        # Skip files that don't have span annotation columns (e.g. avg_seg_scores files)
        if "severity" not in raw.columns or "hypothesis" not in raw.columns:
            logger.debug(f"Skipping {f.name} — missing severity or hypothesis column")
            continue

        # Map severity: neutral/no-error rows become no_error
        raw["severity"] = raw["severity"].str.strip().str.lower()
        raw["error_type"] = (
            raw["error_type"].str.strip().str.lower()
            .str.replace(r"^[^/]+/", "", regex=True)  # "Accuracy/Mistranslation" → "mistranslation"
        )
        # Drop rows with garbled severity values (e.g. trailing-tab files where JSON
        # blob shifts into the severity column)
        valid_sev = {"major", "minor", "critical", "neutral", "no-error", "no_error", "hotw-test", ""}
        raw = raw[raw["severity"].isin(valid_sev)].copy()
        if raw.empty:
            continue

        no_error_mask = raw["severity"].isin(["", "neutral", "no_error", "no-error"])
        raw.loc[no_error_mask, "severity"] = "no_error"
        raw.loc[no_error_mask, "error_type"] = "no_error"

        # Build segment_id: use seg_id column if present, else enumerate
        if "seg_id" in raw.columns:
            raw["segment_id"] = raw["lang"] + "_" + raw["seg_id"].astype(str)
        else:
            raw["segment_id"] = raw["lang"] + "_" + raw.index.astype(str)

        # Span character offsets (not in Google TSVs — mark as unknown)
        raw["error_start"] = -1
        raw["error_end"] = -1

        for col in ["source", "hypothesis", "reference"]:
            if col not in raw.columns:
                raw[col] = ""

        raw["annotation_tier"] = AnnotationTier.HUMAN_MQM

        keep = ["segment_id", "source", "hypothesis", "reference", "lang",
                "error_type", "severity", "error_start", "error_end", "annotation_tier"]
        frames.append(raw[[c for c in keep if c in raw.columns]])
        logger.info(f"  Loaded {len(raw)} span rows from {f.name}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
