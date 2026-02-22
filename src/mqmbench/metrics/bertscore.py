"""BERTScore metric.

Uses microsoft/mdeberta-v3-base by default for multilingual coverage.
F1 scores are returned (range roughly [0, 1], higher is better).
"""

from typing import Optional

import bert_score


def score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    lang: Optional[str] = None,
    model_type: str = "microsoft/mdeberta-v3-base",
    batch_size: int = 32,
) -> list[float]:
    """Compute BERTScore F1 for each hypothesis/reference pair.

    Args:
        sources: Source sentences (unused).
        hypotheses: MT output sentences.
        references: Human reference sentences.
        lang: BCP-47 language code — passed to bert_score if model_type is None.
        model_type: HuggingFace model ID. If provided, overrides lang-based selection.
        batch_size: Batch size for encoding.

    Returns:
        List of BERTScore F1 values in approximately [0, 1].
    """
    _, _, f1 = bert_score.score(
        hypotheses,
        references,
        model_type=model_type,
        lang=None if model_type else lang,
        batch_size=batch_size,
        verbose=False,
    )
    return f1.tolist()
