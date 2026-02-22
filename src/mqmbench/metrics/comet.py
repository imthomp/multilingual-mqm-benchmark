"""COMET and xCOMET metrics via unbabel-comet.

COMET (wmt22-comet-da): reference-based, scores in approximately [0, 1].
xCOMET (XCOMET-XL): also provides MQM-style error span detection.
"""

from typing import Optional

from comet import download_model, load_from_checkpoint


def _load_model(model_name: str):
    model_path = download_model(model_name)
    return load_from_checkpoint(model_path)


def score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    lang: Optional[str] = None,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 16,
    gpus: int = 0,
) -> list[float]:
    """Compute COMET scores for each (source, hypothesis, reference) triple.

    Args:
        sources: Source sentences.
        hypotheses: MT output sentences.
        references: Human reference sentences.
        lang: Target language code (unused — COMET is language-agnostic).
        model_name: COMET model to use.
        batch_size: Batch size.
        gpus: Number of GPUs (0 = CPU).

    Returns:
        List of COMET scores, approximately in [0, 1].
    """
    model = _load_model(model_name)
    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return output.scores


def score_xcomet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    lang: Optional[str] = None,
    model_name: str = "Unbabel/XCOMET-XL",
    batch_size: int = 8,
    gpus: int = 0,
) -> list[float]:
    """Compute xCOMET scores (same interface as score(), separate function for model clarity)."""
    return score(
        sources, hypotheses, references,
        lang=lang,
        model_name=model_name,
        batch_size=batch_size,
        gpus=gpus,
    )
