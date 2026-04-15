"""Load FLORES-200 source sentences and reference translations for Tier 2.

Uses openlanguagedata/flores_plus — the modern parquet-based replacement for
the old facebook/flores dataset, which relied on a loading script no longer
supported by the current HuggingFace datasets library.
"""

import logging

import pandas as pd
from datasets import load_dataset

from mqmbench.data.utils import ISO_TO_BCP47

logger = logging.getLogger(__name__)

_FLORES_DATASET = "openlanguagedata/flores_plus"


def load_flores(lang_codes: list[str], split: str = "devtest") -> pd.DataFrame:
    """Load FLORES-200 English sources and target references.

    Args:
        lang_codes: 2-letter ISO codes (e.g. ['sw', 'ht', 'lo']).
                    Must be present in data/utils.ISO_TO_BCP47.
        split: Dataset split to use ('devtest' has 1012 segments).

    Returns:
        DataFrame with columns: segment_id, lang, source, reference.
        lang column uses the same 2-letter ISO codes passed in.
    """
    logger.info(f"Loading FLORES+ {split} for languages: {lang_codes}")

    en_texts = load_dataset(_FLORES_DATASET, "eng_Latn", split=split)["text"]

    rows = []
    for iso_code in lang_codes:
        flores_code = ISO_TO_BCP47.get(iso_code)
        if flores_code is None:
            logger.warning(
                f"No FLORES+ BCP-47 code for '{iso_code}'. "
                f"Add it to ISO_TO_BCP47 in data/utils.py. Skipping."
            )
            continue
        try:
            tgt_texts = load_dataset(_FLORES_DATASET, flores_code, split=split)["text"]
            for idx, (src, ref) in enumerate(zip(en_texts, tgt_texts)):
                rows.append({
                    "segment_id": f"flores_{split}_{idx:04d}",
                    "lang": iso_code,
                    "source": src,
                    "reference": ref,
                })
        except Exception as e:
            logger.error(f"Could not load FLORES+ for {iso_code} ({flores_code}): {e}")

    return pd.DataFrame(rows)
