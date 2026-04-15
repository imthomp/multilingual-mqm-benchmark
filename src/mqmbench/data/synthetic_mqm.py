"""Generate synthetic span-level MQM annotations using a local LLM.

Uses GEMBA-MQM style three-shot prompting via a HuggingFace text-generation
model. Designed to run on the BYU supercomputer with HF_HUB_OFFLINE=1.
Results are cached to JSONL files so the model only runs once per language.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from transformers import pipeline as hf_pipeline

from mqmbench.constants import AnnotationTier
from mqmbench.data.utils import read_jsonl_cache, write_jsonl_cache

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a strict professional translation reviewer. Even fluent translations contain minor errors — word choice, register, grammar, or awkward phrasing. Your job is to find them.

Evaluate the translation from English to {target_lang}.

=== EXAMPLES ===

Source: The patient must take the medication twice daily with food.
Translation: Le patient devra prendre le médicament deux fois par jour.
Review:
ERROR: [severity: minor] [category: omission] [span: par jour]

Source: Scientists discovered a new species of deep-sea fish near the Mariana Trench.
Translation: Los científicos descubrieron una nueva especie de pez de aguas profundas cerca de la Fosa de las Marianas.
Review:
NO ERRORS

Source: The contract will expire at the end of the fiscal year unless renewed.
Translation: Il contratto scadrà alla fine dell anno fiscale se non rinnovato.
Review:
ERROR: [severity: minor] [category: grammar] [span: dell anno]

=== YOUR TASK ===

Source: {source}
Translation: {hypothesis}

Review (identify ALL errors, or output NO ERRORS if truly none):"""

_ERROR_PATTERN = re.compile(
    r"ERROR:\s*\[severity:\s*(major|minor)\]\s*"
    r"\[category:\s*([^\]]+)\]\s*\[span:\s*([^\]]+)\]",
    re.IGNORECASE,
)


def _parse_output(text: str, base_info: dict) -> list[dict]:
    """Parse LLM output into span-level annotation rows."""
    if "NO ERRORS" in text.upper():
        row = dict(base_info)
        row.update({"error_type": "no_error", "severity": "no_error",
                    "error_start": -1, "error_end": -1})
        return [row]

    rows = []
    for match in _ERROR_PATTERN.finditer(text):
        sev, cat, span = match.groups()
        hyp = base_info["hypothesis"]
        start_idx = hyp.find(span.strip())
        end_idx = start_idx + len(span.strip()) if start_idx != -1 else -1
        row = dict(base_info)
        row.update({
            "severity": sev.strip().lower(),
            "error_type": cat.strip().lower(),
            "error_start": start_idx,
            "error_end": end_idx,
        })
        rows.append(row)

    if not rows:
        row = dict(base_info)
        row.update({"error_type": "no_error", "severity": "no_error",
                    "error_start": -1, "error_end": -1})
        rows.append(row)

    return rows


def generate_synthetic_mqm(
    segments_df: pd.DataFrame,
    model_name: str,
    cache_dir: str = "data/cache",
    batch_size: int = 4,
) -> pd.DataFrame:
    """Generate GEMBA-MQM style annotations using a local HuggingFace LLM.

    Args:
        segments_df: DataFrame with columns: segment_id, source, hypothesis,
                     reference, lang.
        model_name: HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct').
                    Must be pre-downloaded; run with HF_HUB_OFFLINE=1 on compute nodes.
        cache_dir: Directory for cached JSONL annotation files.
        batch_size: Prompts per forward pass.

    Returns:
        DataFrame with span-level annotation rows compatible with converter.py.
        Tagged with annotation_tier = AnnotationTier.SYNTHETIC_MQM.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    langs = segments_df["lang"].unique().tolist()
    cache_files = {lang: cache_path / f"synthetic_mqm_{lang}.jsonl" for lang in langs}
    langs_to_annotate = [lang for lang in langs if not cache_files[lang].exists()]

    span_rows = []

    for lang in langs:
        if cache_files[lang].exists():
            logger.info(f"Loading cached synthetic MQM for {lang}")
            span_rows.extend(read_jsonl_cache(cache_files[lang]))

    if langs_to_annotate:
        logger.info(f"Loading local LLM for synthetic MQM: {model_name}")
        pipe = hf_pipeline("text-generation", model=model_name,
                           max_new_tokens=256, do_sample=False)

        for lang in langs_to_annotate:
            group = segments_df[segments_df["lang"] == lang]
            logger.info(f"Generating synthetic MQM for {lang} ({len(group)} segments)...")
            rows_list = group.to_dict(orient="records")

            prompts = [
                PROMPT_TEMPLATE.format(
                    target_lang=lang,
                    source=row["source"],
                    hypothesis=row["hypothesis"],
                )
                for row in rows_list
            ]

            outputs = []
            for i in range(0, len(prompts), batch_size):
                outputs.extend(pipe(prompts[i : i + batch_size]))

            lang_rows = []
            for row, out in zip(rows_list, outputs):
                base_info = {
                    "segment_id": row["segment_id"],
                    "source": row["source"],
                    "hypothesis": row["hypothesis"],
                    "reference": row["reference"],
                    "lang": lang,
                    "annotation_tier": AnnotationTier.SYNTHETIC_MQM,
                }
                lang_rows.extend(_parse_output(out[0]["generated_text"], base_info))

            write_jsonl_cache(cache_files[lang], lang_rows)
            span_rows.extend(lang_rows)

    return pd.DataFrame(span_rows)
