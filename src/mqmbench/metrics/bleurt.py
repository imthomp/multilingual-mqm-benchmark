"""BLEURT metric via HuggingFace PyTorch (no TensorFlow required).

Uses Elron/bleurt-large-512 — a BertForSequenceClassification regression model
trained on BLEURT's WMT rating data. Input order: reference then hypothesis
(BLEURT is reference-first, unlike most other metrics).

Output scores are in approximately [-1, 2]; higher = better quality.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}


def _load(model_name: str):
    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading BLEURT model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


def score(
    hypotheses: list[str],
    references: list[str],
    model_name: str = "Elron/bleurt-large-512",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> list[float]:
    """Compute BLEURT scores for each (hypothesis, reference) pair.

    Note: BLEURT takes reference as first argument, hypothesis as second
    in its tokenizer call — the order here follows our pipeline convention
    (hypotheses first) but is swapped internally.

    Args:
        hypotheses: MT output sentences.
        references: Human reference translations.
        model_name: HuggingFace model ID (must be a BLEURT-style regression model).
        batch_size: Tokenization + inference batch size.
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        List of float scores (higher = better, approx. range -1 to 2).
    """
    tokenizer, model = _load(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    scores = []
    for i in range(0, len(hypotheses), batch_size):
        batch_hyps = hypotheses[i: i + batch_size]
        batch_refs = references[i: i + batch_size]
        # BLEURT: tokenize as (reference, hypothesis)
        inputs = tokenizer(
            batch_refs, batch_hyps,
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
        scores.extend(logits.cpu().tolist())

    return scores
