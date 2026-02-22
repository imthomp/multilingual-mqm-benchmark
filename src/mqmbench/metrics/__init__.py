"""MT metric wrappers.

Each module exposes a single function:
    score(sources, hypotheses, references, lang) -> list[float]

All scores are in [0, 1] or otherwise higher-is-better unless noted.
"""
