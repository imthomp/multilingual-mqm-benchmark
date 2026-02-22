"""ChrF++ score via sacrebleu.

ChrF++ is character n-gram F-score with word n-grams (word_order=2).
Score range: [0, 100] → normalized to [0, 1] here.
"""

from typing import Optional

import sacrebleu


def score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    lang: Optional[str] = None,
) -> list[float]:
    """Compute sentence-level ChrF++ for each hypothesis/reference pair.

    Args:
        sources: Source sentences (unused).
        hypotheses: MT output sentences.
        references: Human reference sentences.
        lang: Target language code (unused).

    Returns:
        List of ChrF++ scores in [0, 1].
    """
    chrf = sacrebleu.CHRF(word_order=2)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        result = chrf.sentence_score(hyp, [ref])
        scores.append(result.score / 100.0)
    return scores
