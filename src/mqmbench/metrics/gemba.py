"""GEMBA-MQM metric.

GEMBA-MQM uses an LLM to directly predict MQM error annotations.
Reference: Kocmi & Federmann (2023) — https://arxiv.org/abs/2303.12402

The LLM is prompted with the source + hypothesis and asked to identify
errors by category and severity, then a penalty score is computed.

This implementation uses a local HuggingFace model (no API key required),
suited for running on the BYU supercomputer with HF_HUB_OFFLINE=1.
"""

import re
from typing import Optional

from transformers import pipeline as hf_pipeline

GEMBA_PROMPT_TEMPLATE = """You are evaluating a machine translation from {source_lang} to {target_lang}.

Source: {source}
Translation: {hypothesis}

List any translation errors in the format:
ERROR: [severity: major|minor] [category: mistranslation|omission|grammar|register|other] [text: ...]

If there are no errors, write: NO ERRORS"""

SEVERITY_PENALTIES = {"major": 5.0, "minor": 1.0}


def _parse_gemba_output(text: str) -> float:
    """Parse LLM output and compute MQM penalty score."""
    if "NO ERRORS" in text.upper():
        return 0.0
    penalty = 0.0
    for match in re.finditer(r"severity:\s*(major|minor)", text, re.IGNORECASE):
        sev = match.group(1).lower()
        penalty += SEVERITY_PENALTIES.get(sev, 0.0)
    return penalty


def score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    lang: Optional[str] = None,
    model_name: str = "",
    source_lang: str = "English",
    target_lang: str = "the target language",
    batch_size: int = 4,
) -> list[float]:
    """Compute GEMBA-MQM scores using a local LLM.

    Args:
        sources: Source sentences.
        hypotheses: MT output sentences.
        references: Reference sentences (unused — GEMBA is reference-free).
        lang: Target language name or code (used in prompt).
        model_name: HuggingFace model ID for the LLM.
        source_lang: Human-readable source language name for prompt.
        target_lang: Human-readable target language name for prompt.
        batch_size: Number of prompts to process at once.

    Returns:
        List of MQM penalty scores (0 = perfect, higher = more errors).
        NOTE: Lower is better, unlike other metrics. Caller should negate or
        convert to quality_score = 1 / (1 + penalty) for correlation.
    """
    if not model_name:
        raise ValueError(
            "gemba.score() requires model_name. "
            "Set metrics.gemba_model in settings.toml."
        )

    if lang:
        target_lang = lang

    pipe = hf_pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=256,
        do_sample=False,
    )

    prompts = [
        GEMBA_PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=src,
            hypothesis=hyp,
        )
        for src, hyp in zip(sources, hypotheses)
    ]

    penalties = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs = pipe(batch)
        for out in outputs:
            generated = out[0]["generated_text"]
            penalties.append(_parse_gemba_output(generated))

    return penalties
