"""BLEU score via sacrebleu.

Returns corpus-level BLEU decomposed to sentence-level using sacrebleu's
sentence_bleu, which applies add-1 smoothing by default.
Score range: [0, 1] (higher is better).
"""

from typing import Optional

import sacrebleu


def score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    lang: Optional[str] = None,
) -> list[float]:
    """Compute sentence-level BLEU for each hypothesis/reference pair.

    Args:
        sources: Source sentences (unused by BLEU, kept for uniform interface).
        hypotheses: MT output sentences.
        references: Human reference sentences.
        lang: Target language code (unused, kept for uniform interface).

    Returns:
        List of BLEU scores in [0, 100] divided by 100 → [0, 1].
    """
    scores = []
    for hyp, ref in zip(hypotheses, references):
        result = sacrebleu.sentence_bleu(hyp, [ref])
        scores.append(result.score / 100.0)
    return scores
