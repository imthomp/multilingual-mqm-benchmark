"""Compute Pearson and Spearman correlations between metric scores and human MQM scores.

Primary analysis:
    - Per-language Pearson r and Spearman ρ for each metric vs. human quality score
    - Group by resource tier (high / medium / low)
    - Group by error category (accuracy vs. fluency)

All results are returned as a DataFrame for easy export to CSV or LaTeX.
"""

from typing import Optional

import pandas as pd
from scipy import stats

# Language resource tiers matching settings.toml
RESOURCE_TIERS = {
    "high": ["es", "pt"],
    "medium": ["th", "hr", "tl", "hy"],
    "low": ["ht", "lo", "mh", "nv", "gil", "to"],
}

# MQM error categories grouped into accuracy vs. fluency
ACCURACY_ERRORS = {"mistranslation", "omission", "addition", "untranslated"}
FLUENCY_ERRORS = {"grammar", "spelling", "punctuation", "register", "style"}


def _pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def correlate_metric_vs_human(
    human_scores: list[float],
    metric_scores: list[float],
    metric_name: str,
    lang: str,
) -> dict:
    """Compute Pearson and Spearman for one metric against human scores."""
    if len(human_scores) < 3:
        return {
            "lang": lang,
            "metric": metric_name,
            "n": len(human_scores),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
        }
    pr, pp = _pearson(human_scores, metric_scores)
    sr, sp = _spearman(human_scores, metric_scores)
    return {
        "lang": lang,
        "metric": metric_name,
        "n": len(human_scores),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    }


def run_correlation_analysis(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    resource_tiers: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Run full correlation analysis across all languages and metrics.

    Args:
        scores_df: DataFrame with one row per segment, containing human score
                   column, metric score columns, and a 'lang' column.
        metric_columns: Column names of metric scores to evaluate.
        human_column: Column name of human quality scores.
        resource_tiers: Dict mapping tier name → list of lang codes.

    Returns:
        DataFrame with rows: (lang, resource_tier, metric, n, pearson_r, pearson_p,
                               spearman_r, spearman_p)
    """
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS

    lang_to_tier = {
        lang: tier
        for tier, langs in resource_tiers.items()
        for lang in langs
    }

    rows = []
    for lang, group in scores_df.groupby("lang"):
        human = group[human_column].tolist()
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            metric_vals = group[metric].tolist()
            result = correlate_metric_vs_human(human, metric_vals, metric, str(lang))
            result["resource_tier"] = lang_to_tier.get(str(lang), "unknown")
            rows.append(result)

    result_df = pd.DataFrame(rows, columns=[
        "lang", "resource_tier", "metric", "n",
        "pearson_r", "pearson_p", "spearman_r", "spearman_p",
    ])
    return result_df.sort_values(["metric", "resource_tier", "lang"])


def summarize_by_tier(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-language correlations to per-tier averages."""
    return (
        correlation_df
        .groupby(["metric", "resource_tier"])[["pearson_r", "spearman_r"]]
        .mean()
        .reset_index()
        .sort_values(["metric", "resource_tier"])
    )
