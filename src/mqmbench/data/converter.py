"""Convert span-level MQM annotations to sentence-level scores.

Penalty weights follow the WMT MQM shared task convention:
    critical error → 25 penalty points  (non-translation / complete failure)
    major error    →  5 penalty points
    minor error    →  1 penalty point
    neutral        →  0 penalty points

Sentence score formula:
    raw_score  = -sum(penalty_i for each error in segment)
    normalized = raw_score / num_words(hypothesis)  [penalty per word]

The normalized score is negative (0 = perfect). We also produce a [0, 1]
quality score for easier interpretation:
    quality = 1 / (1 + error_penalty)   [1 = perfect, decreases with errors]
"""

import pandas as pd

# WMT MQM penalty weights
SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 25.0,
    "major":     5.0,
    "minor":     1.0,
    "neutral":   0.0,
    "no_error":  0.0,
}


def _word_count(text: str) -> int:
    return max(len(str(text).split()), 1)


def annotations_to_sentence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate span-level error rows into one score per segment.

    Args:
        df: DataFrame from loader.load_annotations() with one row per error span.

    Returns:
        DataFrame with one row per segment containing:
            segment_id, source, hypothesis, reference, lang,
            error_penalty  (raw sum of MQM penalty points),
            normalized_score  (penalty per word, <= 0),
            quality_score  (0-1, 1 = perfect translation),
            num_errors, major_errors, minor_errors
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "segment_id", "source", "hypothesis", "reference", "lang",
            "error_penalty", "normalized_score", "quality_score",
            "num_errors", "critical_errors", "major_errors", "minor_errors",
        ])

    unknown_severities = set(df["severity"].unique()) - set(SEVERITY_WEIGHTS)
    if unknown_severities:
        raise ValueError(
            f"Unknown severity values: {unknown_severities}. "
            f"Expected one of: {list(SEVERITY_WEIGHTS)}"
        )

    df = df.copy()
    df["penalty"] = df["severity"].map(SEVERITY_WEIGHTS)
    df["is_critical"] = (df["severity"] == "critical").astype(int)
    df["is_major"] = (df["severity"] == "major").astype(int)
    df["is_minor"] = (df["severity"] == "minor").astype(int)
    df["is_error"] = (df["severity"].isin(["critical", "major", "minor"])).astype(int)

    grouped = df.groupby(
        ["segment_id", "source", "hypothesis", "reference", "lang"],
        as_index=False,
        sort=False,
    ).agg(
        error_penalty=("penalty", "sum"),
        num_errors=("is_error", "sum"),
        critical_errors=("is_critical", "sum"),
        major_errors=("is_major", "sum"),
        minor_errors=("is_minor", "sum"),
    )

    grouped["num_words"] = grouped["hypothesis"].apply(_word_count)
    grouped["normalized_score"] = -(grouped["error_penalty"] / grouped["num_words"])
    grouped["quality_score"] = 1.0 / (1.0 + grouped["error_penalty"])

    grouped = grouped.drop(columns=["num_words"])
    return grouped


def get_sentence_scores(df: pd.DataFrame, lang: str | None = None) -> pd.DataFrame:
    """Convenience wrapper: optionally filter by language before converting."""
    if lang is not None:
        df = df[df["lang"] == lang]
    return annotations_to_sentence_scores(df)
