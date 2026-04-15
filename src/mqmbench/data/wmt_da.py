"""Load WMT Direct Assessment scores from HuggingFace.

Dataset: RicardoRei/wmt-da-human-evaluation
Covers 41+ language pairs. Available pairs are discovered at runtime;
configure wmt_da_target_langs in settings.toml to filter to desired languages.

DA scores are z-scores (mean 0, unit variance per annotator). They are
normalized per-language to [0, 1] for compatibility with the unified
quality_score column. Since all correlation measures (Pearson/Spearman/Kendall)
are rank-invariant, the normalization does not affect results.
"""

import logging

import pandas as pd
from datasets import load_dataset

from mqmbench.constants import AnnotationTier
from mqmbench.data.utils import build_segment_id, lang_from_pair, rename_wmt_columns

logger = logging.getLogger(__name__)


def _normalize_per_lang(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize DA scores to [0, 1] within each language."""
    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

    df = df.copy()
    df["quality_score"] = df.groupby("lang")["da_score"].transform(_minmax)
    return df


def load_wmt_da(target_langs: list[str] | None = None) -> pd.DataFrame:
    """Load Tier 1b human Direct Assessment sentence-level scores.

    Args:
        target_langs: 2-letter ISO codes to filter for (e.g. ['hr', 'ro']).
                      If None, loads all available languages.

    Returns:
        DataFrame with one row per segment, columns:
            segment_id, source, hypothesis, reference, lang,
            da_score, quality_score, annotation_tier.
        Does NOT pass through converter.py (no span annotations).
    """
    logger.info("Loading WMT DA dataset from HuggingFace...")
    df = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train").to_pandas()

    df["lang"] = df["lp"].apply(lang_from_pair)

    available = sorted(df["lang"].unique().tolist())
    logger.info(f"WMT DA available target languages: {available}")

    if target_langs:
        missing = [l for l in target_langs if l not in available]
        if missing:
            logger.warning(f"Requested DA languages not in dataset: {missing}")
        df = df[df["lang"].isin(target_langs)].copy()

    if df.empty:
        logger.warning("No DA data matched the requested languages.")
        return pd.DataFrame(columns=[
            "segment_id", "source", "hypothesis", "reference",
            "lang", "da_score", "quality_score", "annotation_tier",
        ])

    df = rename_wmt_columns(df)

    if "score" in df.columns and "da_score" not in df.columns:
        df = df.rename(columns={"score": "da_score"})

    df["segment_id"] = build_segment_id(df)
    df = _normalize_per_lang(df)
    df["annotation_tier"] = AnnotationTier.HUMAN_DA

    keep = ["segment_id", "source", "hypothesis", "reference",
            "lang", "da_score", "quality_score", "annotation_tier", "domain"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
