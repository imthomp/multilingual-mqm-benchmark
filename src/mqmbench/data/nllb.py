"""Generate MT hypotheses for Tier 2 using NLLB-200.

Translates FLORES-200 English source sentences into target languages.
Results are cached per language to avoid re-running the model.
Designed for the BYU supercomputer (HF_HUB_OFFLINE=1, GPU available).
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from mqmbench.data.utils import ISO_TO_BCP47, read_jsonl_cache, write_jsonl_cache

logger = logging.getLogger(__name__)


def generate_hypotheses(
    segments_df: pd.DataFrame,
    model_name: str = "facebook/nllb-200-distilled-600M",
    cache_dir: str = "data/cache",
    batch_size: int = 16,
) -> pd.DataFrame:
    """Translate source text to target languages using NLLB-200.

    Args:
        segments_df: DataFrame from load_flores() with columns:
                     segment_id, lang, source, reference.
                     lang must be a 2-letter ISO code present in ISO_TO_BCP47.
        model_name: HuggingFace NLLB model ID.
        cache_dir: Directory for cached JSONL hypothesis files.
        batch_size: Sentences per translation batch.

    Returns:
        Input DataFrame with an added 'hypothesis' column.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    langs = segments_df["lang"].unique().tolist()
    cache_files = {lang: cache_path / f"nllb_hypotheses_{lang}.jsonl" for lang in langs}
    langs_to_translate = [lang for lang in langs if not cache_files[lang].exists()]

    results = []

    # Load cached results first without touching the model
    for lang in langs:
        if cache_files[lang].exists():
            logger.info(f"Loading cached NLLB hypotheses for {lang}")
            results.extend(read_jsonl_cache(cache_files[lang]))

    if langs_to_translate:
        logger.info(f"Loading NLLB-200 model: {model_name}")
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for lang_code in langs_to_translate:
            nllb_lang = ISO_TO_BCP47.get(lang_code)
            if nllb_lang is None:
                logger.warning(
                    f"No BCP-47 code for '{lang_code}'. "
                    f"Add it to ISO_TO_BCP47 in data/utils.py. Skipping."
                )
                continue

            group = segments_df[segments_df["lang"] == lang_code]
            logger.info(f"Translating {len(group)} segments to {lang_code} ({nllb_lang})...")

            translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang="eng_Latn",
                tgt_lang=nllb_lang,
                device=device,
                batch_size=batch_size,
            )

            lang_results = []
            for row, out in zip(
                group.to_dict(orient="records"),
                translator(group["source"].tolist(), max_length=256),
            ):
                row["hypothesis"] = out["translation_text"]
                lang_results.append(row)

            write_jsonl_cache(cache_files[lang_code], lang_results)
            results.extend(lang_results)

    return pd.DataFrame(results)
