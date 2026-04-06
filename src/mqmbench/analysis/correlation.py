"""Compute Pearson, Spearman, and Kendall correlations.

Primary analysis:
    - Per-language metrics vs. human quality score (all tiers)
    - Group by resource tier (high / medium / low)
    - Group by language family and script type (per Dr. Fulda's suggestion)
    - Group by error category (accuracy vs. fluency) (Tier 1a only)
"""

from typing import Optional
import pandas as pd
from scipy import stats

from mqmbench.constants import AnnotationTier

RESOURCE_TIERS = {
    "high":   ["de", "zh", "ru", "he"],
    "medium": ["es", "cs", "tr", "uk"],
    "low":    ["ha", "km", "ps", "sw", "ht", "lo"],
}

# Language family groupings for secondary analysis
LANGUAGE_FAMILIES = {
    "indo_european": ["de", "ru", "es", "cs", "uk", "ps"],  # ps = Iranian branch
    "afro_asiatic":  ["he", "ha"],   # he = Semitic, ha = Chadic
    "sino_tibetan":  ["zh"],
    "turkic":        ["tr"],
    "austroasiatic": ["km"],
    "tai_kadai":     ["lo"],
    "niger_congo":   ["sw"],
    "creole":        ["ht"],
}

# Script type per Dr. Fulda's suggestion: logographic vs. phonographic
SCRIPT_TYPES = {
    "logographic": ["zh"],                                  # character = morpheme/word
    "alphabetic":  ["de", "ru", "es", "cs", "uk", "tr", "sw", "ha", "ht"],  # true alphabets
    "abjad":       ["he", "ps"],                            # consonant-primary (Hebrew, Arabic)
    "abugida":     ["km", "lo"],                            # Brahmic-derived (Khmer, Lao)
}

ACCURACY_ERRORS = {"mistranslation", "omission", "addition", "untranslated"}
FLUENCY_ERRORS = {"grammar", "spelling", "punctuation", "register", "style"}


def _pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _kendall(x: list[float], y: list[float]) -> tuple[float, float]:
    tau, p = stats.kendalltau(x, y)
    return float(tau), float(p)


def correlate_metric_vs_human(
    human_scores: list[float],
    metric_scores: list[float],
    metric_name: str,
    lang: str,
) -> dict:
    """Compute Pearson, Spearman, and Kendall for one metric against human scores."""
    if len(human_scores) < 3:
        return {
            "lang": lang, "metric": metric_name, "n": len(human_scores),
            "pearson_r": None, "pearson_p": None,
            "spearman_r": None, "spearman_p": None,
            "kendall_tau": None, "kendall_p": None,
        }
    pr, pp = _pearson(human_scores, metric_scores)
    sr, sp = _spearman(human_scores, metric_scores)
    kr, kp = _kendall(human_scores, metric_scores)
    return {
        "lang": lang, "metric": metric_name, "n": len(human_scores),
        "pearson_r": pr, "pearson_p": pp,
        "spearman_r": sr, "spearman_p": sp,
        "kendall_tau": kr, "kendall_p": kp,
    }


def run_correlation_analysis(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    resource_tiers: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Run full correlation analysis across all languages and metrics."""
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS

    lang_to_tier = {lang: tier for tier, langs in resource_tiers.items() for lang in langs}
    lang_to_family = {lang: fam for fam, langs in LANGUAGE_FAMILIES.items() for lang in langs}
    lang_to_script = {lang: stype for stype, langs in SCRIPT_TYPES.items() for lang in langs}

    rows = []
    for (lang, tier_name), group in scores_df.groupby(["lang", "annotation_tier"]):
        human = group[human_column].tolist()
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            result = correlate_metric_vs_human(human, group[metric].tolist(), metric, str(lang))
            result["resource_tier"] = lang_to_tier.get(str(lang), "unknown")
            result["annotation_tier"] = tier_name
            result["language_family"] = lang_to_family.get(str(lang), "unknown")
            result["script_type"] = lang_to_script.get(str(lang), "unknown")
            rows.append(result)

    result_df = pd.DataFrame(rows, columns=[
        "lang", "resource_tier", "annotation_tier", "language_family", "script_type",
        "metric", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p",
        "kendall_tau", "kendall_p",
    ])
    return result_df.sort_values(["metric", "resource_tier", "lang"])


def run_category_correlation(
    raw_span_df: pd.DataFrame,
    converter_func,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Run correlation isolated by error category (Accuracy vs. Fluency).

    Args:
        raw_span_df: Unconverted DataFrame with raw span annotations; must
                     include error_type and the metric score columns.
        converter_func: annotations_to_sentence_scores from converter.py.
        metric_columns: Metric column names to evaluate.

    Returns:
        DataFrame like run_correlation_analysis() output plus error_category column.
    """
    mqm_df = raw_span_df[raw_span_df["annotation_tier"] == AnnotationTier.HUMAN_MQM].copy()

    results = []
    for category_name, error_set in [
        ("Accuracy", ACCURACY_ERRORS),
        ("Fluency", FLUENCY_ERRORS),
    ]:
        filtered_spans = mqm_df[mqm_df["error_type"].str.lower().isin(error_set)]
        cat_scores_df = converter_func(filtered_spans)
        cat_scores_df["annotation_tier"] = AnnotationTier.HUMAN_MQM

        if "segment_id" in cat_scores_df.columns and metric_columns:
            metric_lookup = (
                mqm_df[["segment_id"] + [c for c in metric_columns if c in mqm_df.columns]]
                .drop_duplicates("segment_id")
            )
            cat_scores_df = cat_scores_df.merge(metric_lookup, on="segment_id", how="left",
                                                 suffixes=("", "_dup"))

        corr_df = run_correlation_analysis(cat_scores_df, metric_columns)
        corr_df["error_category"] = category_name
        results.append(corr_df)

    return pd.concat(results, ignore_index=True)


def summarize_by_tier(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-language correlations to per-tier averages."""
    return (
        correlation_df
        .groupby(["metric", "resource_tier", "annotation_tier"])[["pearson_r", "spearman_r", "kendall_tau"]]
        .mean()
        .reset_index()
        .sort_values(["metric", "resource_tier"])
    )


def summarize_by_script(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-language correlations by script type."""
    return (
        correlation_df
        .groupby(["metric", "script_type"])[["pearson_r", "spearman_r", "kendall_tau"]]
        .mean()
        .reset_index()
        .sort_values(["metric", "script_type"])
    )


def summarize_by_family(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-language correlations by language family."""
    return (
        correlation_df
        .groupby(["metric", "language_family"])[["pearson_r", "spearman_r", "kendall_tau"]]
        .mean()
        .reset_index()
        .sort_values(["metric", "language_family"])
    )
