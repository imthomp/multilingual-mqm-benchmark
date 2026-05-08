"""Compute Pearson, Spearman, Kendall, and pairwise accuracy correlations.

Primary analysis:
    - Per-language metrics vs. human quality score (all tiers)
    - Group by resource tier (high / medium / low)
    - Group by language family and script type (per Dr. Fulda's suggestion)
    - Group by error category (accuracy vs. fluency) (Tier 1a only)
    - Williams test for significance when comparing two metrics
    - Pairwise accuracy (WMT 2023-2024 primary meta-evaluation measure)
    - Error analysis: segments where metric diverges most from human judgment
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from mqmbench.constants import AnnotationTier

RESOURCE_TIERS = {
    "high":   ["de", "zh", "ru", "he"],
    "medium": ["es", "cs", "tr", "uk", "fr", "pl", "fi", "et", "is", "lt", "lv",
               "bn", "hi", "gu", "ta", "ja", "kk", "xh", "zu"],
    "low":    ["ha", "km", "ps", "sw", "ht", "lo", "th", "my", "am", "ka"],
}

# Language family groupings for secondary analysis
LANGUAGE_FAMILIES = {
    "indo_european": ["de", "ru", "es", "cs", "uk", "ps",   # Germanic/Slavic/Romance/Iranian
                      "fr", "pl", "is", "lt", "lv",          # Romance/Baltic/Nordic
                      "bn", "hi", "gu"],                      # Indo-Aryan
    "afro_asiatic":  ["he", "ha", "am"],  # Semitic (he, am) + Chadic (ha)
    "sino_tibetan":  ["zh", "my"],        # Sinitic + Tibeto-Burman (Burmese)
    "turkic":        ["tr", "kk"],        # Turkish + Kazakh
    "austroasiatic": ["km"],
    "tai_kadai":     ["lo", "th"],        # Lao + Thai
    "niger_congo":   ["sw", "xh", "zu"],  # Bantu family
    "creole":        ["ht"],
    "dravidian":     ["ta"],
    "japonic":       ["ja"],
    "uralic":        ["fi", "et"],
    "kartvelian":    ["ka"],              # Georgian
}

# Script type per Dr. Fulda's suggestion: logographic vs. phonographic
SCRIPT_TYPES = {
    "logographic": ["zh", "ja"],                              # character = morpheme/word; ja = kanji+kana
    "alphabetic":  ["de", "ru", "es", "cs", "uk", "tr", "sw", "ha", "ht",
                    "fr", "pl", "is", "lt", "lv", "kk",
                    "xh", "zu", "fi", "et", "ka"],            # Latin/Cyrillic + Georgian
    "abjad":       ["he", "ps"],                              # consonant-primary (Hebrew, Arabic)
    "abugida":     ["km", "lo", "bn", "hi", "gu", "ta",
                    "th", "my", "am"],                        # Brahmic-derived + Thai/Myanmar/Ge'ez
}

ACCURACY_ERRORS = {"mistranslation", "omission", "addition", "untranslated"}
FLUENCY_ERRORS = {"grammar", "spelling", "punctuation", "register", "style"}

# Morphological type classification (structural property, independent of script/family).
# isolating   = minimal inflection, meaning via word order (zh, th, lo, vi)
# agglutinative = transparent suffixes/prefixes stack predictably (tr, fi, et, ka, kk, sw,
#                 km, my, am, ja, ta — ja is agglutinative in its verbal morphology)
# fusional     = inflection fuses multiple features per morpheme (de, ru, es, cs, fr, pl,
#                uk, is, lt, lv, he, ps, bn, hi, gu, xh, zu, ha, ht)
MORPHOLOGY_TYPES = {
    "isolating":    ["zh", "th", "lo"],
    "agglutinative": ["tr", "fi", "et", "ka", "kk", "sw", "km", "my", "am", "ja", "ta"],
    "fusional":     ["de", "ru", "es", "cs", "fr", "pl", "uk", "is", "lt", "lv",
                     "he", "ps", "bn", "hi", "gu", "xh", "zu", "ha", "ht"],
}

# WMT language pair direction: X→en vs en→X.
# Matters because COMET/BERTScore were trained predominantly on en→X data.
TRANSLATION_DIRECTIONS = {
    "x_to_en": ["zh", "he"],    # zh-en, he-en
    "en_to_x": ["de", "ru", "es", "cs", "tr", "uk", "fr", "pl", "fi", "et", "is",
                "lt", "lv", "bn", "hi", "gu", "ta", "ja", "kk", "xh", "zu",
                "ha", "km", "ps", "sw", "ht", "lo", "th", "my", "am", "ka"],
}


def _pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _kendall(x: list[float], y: list[float]) -> tuple[float, float]:
    tau, p = stats.kendalltau(x, y)
    return float(tau), float(p)


def _bootstrap_ci(
    x: list[float],
    y: list[float],
    stat_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for stat_fn(x, y)."""
    rng = np.random.default_rng(seed)
    arr_x = np.array(x)
    arr_y = np.array(y)
    n = len(arr_x)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_stats.append(stat_fn(arr_x[idx], arr_y[idx]))
        except Exception:
            pass
    if not boot_stats:
        return float("nan"), float("nan")
    lo = float(np.percentile(boot_stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_stats, (1 + ci) / 2 * 100))
    return lo, hi


def pairwise_accuracy(
    human_scores: list[float],
    metric_scores: list[float],
    max_n: int = 8000,
    seed: int = 42,
) -> float:
    """Compute pairwise accuracy (WMT 2023-2024 primary segment-level measure).

    For every pair of segments (i, j), counts how often the metric agrees with
    the human on which translation is better. Tied pairs are excluded from the
    denominator. Returns NaN if fewer than one valid pair.

    For large datasets (n > max_n), draws a random sample to avoid O(n²) memory.
    At max_n=8000, the estimate is stable (>31M pairs evaluated).
    """
    h = np.array(human_scores, dtype=float)
    m = np.array(metric_scores, dtype=float)
    n = len(h)
    if n < 2:
        return float("nan")

    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        h, m = h[idx], m[idx]
        n = max_n

    # Vectorised upper-triangle only
    h_diff = h[:, None] - h[None, :]
    m_diff = m[:, None] - m[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    h_d = h_diff[mask]
    m_d = m_diff[mask]

    valid = (h_d != 0) & (m_d != 0)
    if valid.sum() == 0:
        return float("nan")

    concordant = ((h_d > 0) == (m_d > 0))[valid].sum()
    return float(concordant / valid.sum())


def soft_pairwise_accuracy(
    human_scores: list[float],
    metric_scores: list[float],
    max_n: int = 8000,
    seed: int = 42,
) -> float:
    """WMT 2024 Soft Pairwise Accuracy (SPA).

    Like pairwise accuracy, but metric-tied pairs contribute 0.5 rather than
    being excluded from the denominator. Only human-tied pairs are excluded.
    This is the segment-level meta-evaluation standard from WMT 2024.

    Reference: Thompson et al., WMT 2024 Metrics Shared Task.
    """
    h = np.array(human_scores, dtype=float)
    m = np.array(metric_scores, dtype=float)
    n = len(h)
    if n < 2:
        return float("nan")

    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        h, m = h[idx], m[idx]
        n = max_n

    h_diff = h[:, None] - h[None, :]
    m_diff = m[:, None] - m[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    h_d = h_diff[mask]
    m_d = m_diff[mask]

    # Exclude human-tied pairs
    valid = h_d != 0
    if valid.sum() == 0:
        return float("nan")

    h_d_v = h_d[valid]
    m_d_v = m_d[valid]
    # Concordant = metric agrees with human direction; tied metric = 0.5
    concordant = (h_d_v > 0) == (m_d_v > 0)
    tied = m_d_v == 0
    scores = np.where(tied, 0.5, concordant.astype(float))
    return float(scores.mean())


def williams_test(
    human_scores: list[float],
    metric1_scores: list[float],
    metric2_scores: list[float],
) -> tuple[float, float]:
    """Williams (1959) test for the significance of the difference between two
    correlated correlations sharing a common criterion (human scores).

    Tests H0: r(metric1, human) == r(metric2, human).

    Returns:
        (t_statistic, p_value) — two-tailed, df = n - 3.
        Returns (nan, nan) if n < 6 or the formula is degenerate.

    Reference: Williams, E.J. (1959). Regression Analysis. Wiley.
               Graham (2003) doi:10.1111/1469-8986.00052
    """
    h = np.array(human_scores, dtype=float)
    m1 = np.array(metric1_scores, dtype=float)
    m2 = np.array(metric2_scores, dtype=float)
    n = len(h)
    if n < 6:
        return float("nan"), float("nan")

    r1h = float(stats.pearsonr(m1, h).statistic)
    r2h = float(stats.pearsonr(m2, h).statistic)
    r12 = float(stats.pearsonr(m1, m2).statistic)

    # Determinant of the 3x3 correlation matrix
    K = 1 - r1h**2 - r2h**2 - r12**2 + 2 * r1h * r2h * r12
    denom_sq = 2 * K / (n - 1)
    if denom_sq <= 0:
        return float("nan"), float("nan")

    t = (r1h - r2h) * np.sqrt((n - 1) * (1 + r12)) / np.sqrt(denom_sq)
    p = float(2 * stats.t.sf(abs(t), df=n - 3))
    return float(t), p


def run_williams_tests(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    reference_metric: str = "comet",
) -> pd.DataFrame:
    """Run Williams test for each metric vs. a reference metric, per language.

    Args:
        scores_df: DataFrame with human and metric score columns.
        metric_columns: All metric columns to compare.
        human_column: Column containing human quality scores.
        reference_metric: Metric to compare all others against (default: comet).

    Returns:
        DataFrame with columns: lang, metric, reference_metric, t, p, significant.
    """
    if reference_metric not in metric_columns:
        return pd.DataFrame()

    rows = []
    for lang, group in scores_df.groupby("lang"):
        human = group[human_column].tolist()
        ref_scores = group[reference_metric].tolist()
        for metric in metric_columns:
            if metric == reference_metric or metric not in group.columns:
                continue
            t, p = williams_test(human, group[metric].tolist(), ref_scores)
            rows.append({
                "lang": lang,
                "metric": metric,
                "reference_metric": reference_metric,
                "n": len(human),
                "t_statistic": t,
                "p_value": p,
                "significant_p05": p < 0.05 if not np.isnan(p) else False,
            })
    return pd.DataFrame(rows)


def analyze_metric_disagreements(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    top_n: int = 50,
) -> pd.DataFrame:
    """Identify segments where metrics disagree most with human judgment.

    For each (lang, metric) pair, computes the residual between normalised
    metric rank and human rank, then returns the top_n worst-disagreement
    segments. Useful for qualitative error analysis.

    Args:
        scores_df: DataFrame with source, hypothesis, human and metric columns.
        metric_columns: Metric columns to analyse.
        human_column: Column containing human quality scores.
        top_n: Number of worst-disagreement segments to return per (lang, metric).

    Returns:
        DataFrame with columns: lang, metric, segment_id, source, hypothesis,
        human_score, metric_score, human_rank_pct, metric_rank_pct,
        rank_disagreement, direction.
        Sorted by |rank_disagreement| descending.
    """
    rows = []
    text_cols = [c for c in ["segment_id", "source", "hypothesis"] if c in scores_df.columns]

    for lang, group in scores_df.groupby("lang"):
        group = group.copy().reset_index(drop=True)
        n = len(group)
        if n < 10:
            continue

        # Human percentile rank (0=worst, 1=best)
        h_rank = group[human_column].rank(pct=True).values

        for metric in metric_columns:
            if metric not in group.columns:
                continue
            m_vals = group[metric]
            if m_vals.isna().all():
                continue
            m_rank = m_vals.rank(pct=True).values
            residual = m_rank - h_rank  # positive = metric over-scores vs. human

            worst_idx = np.argsort(np.abs(residual))[::-1][:top_n]
            for idx in worst_idx:
                row = {
                    "lang": lang,
                    "metric": metric,
                    "human_score": float(group[human_column].iloc[idx]),
                    "metric_score": float(group[metric].iloc[idx]),
                    "human_rank_pct": float(h_rank[idx]),
                    "metric_rank_pct": float(m_rank[idx]),
                    "rank_disagreement": float(residual[idx]),
                    "direction": "over-scored" if residual[idx] > 0 else "under-scored",
                }
                for col in text_cols:
                    row[col] = group[col].iloc[idx]
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out_cols = ["lang", "metric"] + text_cols + [
        "human_score", "metric_score",
        "human_rank_pct", "metric_rank_pct",
        "rank_disagreement", "direction",
    ]
    result = pd.DataFrame(rows)
    return result[[c for c in out_cols if c in result.columns]].sort_values(
        "rank_disagreement", key=abs, ascending=False
    ).reset_index(drop=True)


def correlate_metric_vs_human(
    human_scores: list[float],
    metric_scores: list[float],
    metric_name: str,
    lang: str,
    n_boot: int = 1000,
) -> dict:
    """Compute Pearson, Spearman, Kendall, pairwise accuracy, and SPA for one
    metric against human scores, with 95% bootstrap CIs for Spearman r."""
    if len(human_scores) < 3:
        return {
            "lang": lang, "metric": metric_name, "n": len(human_scores),
            "pairwise_acc": None, "spa": None,
            "pearson_r": None, "pearson_p": None,
            "spearman_r": None, "spearman_p": None, "spearman_ci_lo": None, "spearman_ci_hi": None,
            "kendall_tau": None, "kendall_p": None,
        }
    pr, pp = _pearson(human_scores, metric_scores)
    sr, sp = _spearman(human_scores, metric_scores)
    kr, kp = _kendall(human_scores, metric_scores)
    pa = pairwise_accuracy(human_scores, metric_scores)
    spa = soft_pairwise_accuracy(human_scores, metric_scores)
    ci_lo, ci_hi = _bootstrap_ci(
        human_scores, metric_scores,
        stat_fn=lambda x, y: stats.spearmanr(x, y).statistic,
        n_boot=n_boot,
    )
    return {
        "lang": lang, "metric": metric_name, "n": len(human_scores),
        "pairwise_acc": pa, "spa": spa,
        "pearson_r": pr, "pearson_p": pp,
        "spearman_r": sr, "spearman_p": sp, "spearman_ci_lo": ci_lo, "spearman_ci_hi": ci_hi,
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
        # Capture dominant domain for this lang (if column present)
        domain = (
            group["domain"].mode().iloc[0]
            if "domain" in group.columns and not group["domain"].isna().all()
            else "unknown"
        )
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            result = correlate_metric_vs_human(human, group[metric].tolist(), metric, str(lang))
            result["resource_tier"] = lang_to_tier.get(str(lang), "unknown")
            result["annotation_tier"] = tier_name
            result["language_family"] = lang_to_family.get(str(lang), "unknown")
            result["script_type"] = lang_to_script.get(str(lang), "unknown")
            result["domain"] = domain
            rows.append(result)

    result_df = pd.DataFrame(rows, columns=[
        "lang", "resource_tier", "annotation_tier", "language_family", "script_type",
        "domain", "metric", "n", "pairwise_acc", "spa", "pearson_r", "pearson_p",
        "spearman_r", "spearman_p", "spearman_ci_lo", "spearman_ci_hi",
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


_SUMMARY_COLS = ["pairwise_acc", "spa", "pearson_r", "spearman_r", "kendall_tau"]


def summarize_by_tier(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-language correlations to per-tier averages."""
    cols = [c for c in _SUMMARY_COLS if c in correlation_df.columns]
    return (
        correlation_df
        .groupby(["metric", "resource_tier", "annotation_tier"])[cols]
        .mean()
        .reset_index()
        .sort_values(["metric", "resource_tier"])
    )


def summarize_by_script(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-language correlations by script type."""
    cols = [c for c in _SUMMARY_COLS if c in correlation_df.columns]
    return (
        correlation_df
        .groupby(["metric", "script_type"])[cols]
        .mean()
        .reset_index()
        .sort_values(["metric", "script_type"])
    )


def summarize_by_family(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-language correlations by language family."""
    cols = [c for c in _SUMMARY_COLS if c in correlation_df.columns]
    return (
        correlation_df
        .groupby(["metric", "language_family"])[cols]
        .mean()
        .reset_index()
        .sort_values(["metric", "language_family"])
    )


def kiwi_vs_comet_by_tier(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Compare COMET-Kiwi variants (reference-free) vs. COMET (reference-based) by language.

    Compares all available kiwi variants (cometkiwi, cometkiwi23) against COMET.
    The key research question: for low-resource languages where references are
    unreliable, does dropping the reference hurt or help? Does the 2023 Kiwi
    close the gap with reference-based COMET?

    Returns:
        DataFrame with per-language columns for COMET + all Kiwi variants plus
        signed advantages. Empty if COMET is absent.
    """
    available_metrics = set(corr_df["metric"].unique())
    kiwi_variants = [m for m in ["cometkiwi", "cometkiwi23"] if m in available_metrics]
    if "comet" not in available_metrics or not kiwi_variants:
        return pd.DataFrame()

    id_cols = ["lang", "resource_tier", "language_family", "script_type"]
    val_cols = ["spearman_r", "pairwise_acc", "spa", "kendall_tau"]
    # Also carry bootstrap CI columns for error bars in plots
    ci_cols = ["spearman_ci_lo", "spearman_ci_hi"]
    val_cols = [c for c in val_cols if c in corr_df.columns]
    ci_cols = [c for c in ci_cols if c in corr_df.columns]

    def _extract(metric_name: str, suffix: str) -> pd.DataFrame:
        all_cols = val_cols + ci_cols
        sub = corr_df[corr_df["metric"] == metric_name][id_cols + all_cols].copy()
        return sub.rename(columns={c: f"{c}_{suffix}" for c in all_cols})

    result = _extract("comet", "comet")
    for variant in kiwi_variants:
        suffix = "kiwi22" if variant == "cometkiwi" else "kiwi23"
        kiwi_df = _extract(variant, suffix)
        result = result.merge(kiwi_df, on=id_cols, how="left")
        for col in val_cols:
            result[f"{suffix}_advantage_{col}"] = result[f"{col}_{suffix}"] - result[f"{col}_comet"]

    return result.sort_values(["resource_tier", "lang"]).reset_index(drop=True)


def run_domain_analysis(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    resource_tiers: Optional[dict[str, list[str]]] = None,
    min_segments: int = 30,
) -> pd.DataFrame:
    """Correlation analysis stratified by domain (e.g. news vs. conversational).

    Computes per-(lang, domain) Spearman r and SPA so we can show that metric
    reliability is stable within domains but drops when domains are pooled —
    addressing the Ukrainian conversational-text confound and providing a
    controlled within-domain estimate.

    Args:
        scores_df: Full scores DataFrame with a 'domain' column.
        metric_columns: Metric columns to evaluate.
        human_column: Human quality score column.
        min_segments: Skip (lang, domain) groups with fewer segments.

    Returns:
        DataFrame like run_correlation_analysis() plus a 'domain' column.
        Empty if 'domain' column is absent.
    """
    if "domain" not in scores_df.columns:
        return pd.DataFrame()
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS

    lang_to_tier = {lang: tier for tier, langs in resource_tiers.items() for lang in langs}
    lang_to_family = {lang: fam for fam, langs in LANGUAGE_FAMILIES.items() for lang in langs}
    lang_to_script = {lang: stype for stype, langs in SCRIPT_TYPES.items() for lang in langs}

    rows = []
    for (lang, domain), group in scores_df.groupby(["lang", "domain"]):
        if len(group) < min_segments or group[human_column].isna().all():
            continue
        human = group[human_column].tolist()
        for metric in metric_columns:
            if metric not in group.columns or group[metric].isna().all():
                continue
            result = correlate_metric_vs_human(human, group[metric].tolist(), metric, str(lang))
            result["resource_tier"] = lang_to_tier.get(str(lang), "unknown")
            result["language_family"] = lang_to_family.get(str(lang), "unknown")
            result["script_type"] = lang_to_script.get(str(lang), "unknown")
            result["domain"] = str(domain)
            rows.append(result)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["domain", "metric", "lang"]).reset_index(drop=True)


def analyze_tier_anomaly(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    resource_tiers: Optional[dict[str, list[str]]] = None,
    human_column: str = "quality_score",
) -> pd.DataFrame:
    """Analyse the medium > high correlation anomaly by decomposing by WMT year.

    For high-resource languages that span multiple WMT years, computes per-year
    Spearman r and compares variance to medium-resource (typically single-year)
    languages. If high-resource languages show high cross-year variance, the
    aggregate correlation is depressed relative to within-year performance.

    Requires a 'year' column in scores_df (present when WMT MQM data is loaded
    with the year field kept, which wmt_mqm.py now does).

    Returns:
        DataFrame with columns: lang, resource_tier, metric, year, n,
        year_spearman_r, aggregate_spearman_r.
        Empty if 'year' column is absent.
    """
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS
    if "year" not in scores_df.columns:
        return pd.DataFrame()

    lang_to_tier = {lang: tier for tier, langs in resource_tiers.items() for lang in langs}
    rows = []

    for lang in scores_df["lang"].unique():
        lang_df = scores_df[scores_df["lang"] == lang].copy()
        tier = lang_to_tier.get(str(lang), "unknown")

        for metric in metric_columns:
            if metric not in lang_df.columns:
                continue

            valid_all = lang_df[[human_column, metric]].dropna()
            if len(valid_all) < 10:
                continue
            agg_r, _ = _spearman(valid_all[human_column].tolist(), valid_all[metric].tolist())

            year_col = lang_df["year"].dropna()
            if year_col.nunique() > 1:
                for yr, yr_group in lang_df.groupby("year"):
                    yr_valid = yr_group[[human_column, metric]].dropna()
                    if len(yr_valid) < 10:
                        continue
                    yr_r, _ = _spearman(yr_valid[human_column].tolist(), yr_valid[metric].tolist())
                    rows.append({
                        "lang": str(lang), "resource_tier": tier, "metric": metric,
                        "year": str(yr), "n": len(yr_valid),
                        "year_spearman_r": yr_r, "aggregate_spearman_r": agg_r,
                    })
            else:
                yr_val = str(year_col.iloc[0]) if len(year_col) else "unknown"
                rows.append({
                    "lang": str(lang), "resource_tier": tier, "metric": metric,
                    "year": yr_val, "n": len(valid_all),
                    "year_spearman_r": agg_r, "aggregate_spearman_r": agg_r,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["resource_tier", "lang", "metric", "year"])


def run_system_level_analysis(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    resource_tiers: Optional[dict[str, list[str]]] = None,
    min_systems: int = 5,
) -> pd.DataFrame:
    """System-level correlation analysis.

    Aggregates metric scores per (lang, system) then correlates system rankings
    with human quality rankings. System-level Spearman r is the primary measure
    used in the WMT Metrics Shared Task. It is typically much higher than
    segment-level, which is itself a key finding: automated metrics are much
    better at ranking MT systems than at scoring individual translations.

    Args:
        scores_df: Segment-level scores; must have a 'system' column.
        metric_columns: Metric columns to evaluate.
        min_systems: Skip languages with fewer distinct systems (unstable correlation).

    Returns:
        DataFrame like run_correlation_analysis() with an extra 'level' = 'system'
        column. Empty if 'system' column is absent.
    """
    if "system" not in scores_df.columns:
        return pd.DataFrame()
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS

    lang_to_tier = {lang: tier for tier, langs in resource_tiers.items() for lang in langs}
    lang_to_family = {lang: fam for fam, langs in LANGUAGE_FAMILIES.items() for lang in langs}
    lang_to_script = {lang: stype for stype, langs in SCRIPT_TYPES.items() for lang in langs}

    rows = []
    for lang, group in scores_df.groupby("lang"):
        n_systems = group["system"].nunique()
        if n_systems < min_systems:
            continue
        sys_df = (
            group.groupby("system")[[human_column] + [m for m in metric_columns if m in group.columns]]
            .mean()
            .reset_index()
        )
        human = sys_df[human_column].tolist()
        domain = (
            group["domain"].mode().iloc[0]
            if "domain" in group.columns and not group["domain"].isna().all()
            else "unknown"
        )
        for metric in metric_columns:
            if metric not in sys_df.columns or sys_df[metric].isna().all():
                continue
            result = correlate_metric_vs_human(human, sys_df[metric].tolist(), metric, str(lang))
            result["resource_tier"] = lang_to_tier.get(str(lang), "unknown")
            result["language_family"] = lang_to_family.get(str(lang), "unknown")
            result["script_type"] = lang_to_script.get(str(lang), "unknown")
            result["domain"] = domain
            result["n_systems"] = n_systems
            result["level"] = "system"
            rows.append(result)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["metric", "resource_tier", "lang"]).reset_index(drop=True)


def compute_metric_correlations(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Pairwise Spearman correlation matrix between all metrics.

    High inter-metric correlation → metrics are redundant for ranking purposes.
    Low inter-metric correlation → complementary signals (ensemble could help).
    Computed over all segments pooled across languages and tiers.

    Returns:
        Square DataFrame of Spearman r values (metric × metric).
    """
    valid_cols = [c for c in metric_columns if c in scores_df.columns]
    if len(valid_cols) < 2:
        return pd.DataFrame()
    return scores_df[valid_cols].corr(method="spearman")


def reference_quality_effect(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    resource_tiers: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Compare COMET vs. COMET-Kiwi advantage split by reference quality tier.

    Compares how much reference-free Kiwi gains (or loses) relative to
    reference-based COMET when references are professional MQM annotations
    vs. crowd-sourced Direct Assessment ratings. If Kiwi's advantage grows
    on DA (noisier references), that supports the hypothesis that reference
    quality is a confounding factor in metric reliability for low-resource langs.

    Returns:
        DataFrame with per-(lang, annotation_tier) advantage values. Empty if
        neither COMET nor any Kiwi variant is available.
    """
    available = set(scores_df.columns)
    kiwi_cols = [m for m in ["cometkiwi", "cometkiwi23"] if m in available]
    if "comet" not in available or not kiwi_cols or human_column not in available:
        return pd.DataFrame()
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS

    lang_to_tier = {lang: tier for tier, langs in resource_tiers.items() for lang in langs}
    rows = []
    for (lang, ann_tier), group in scores_df.groupby(["lang", "annotation_tier"]):
        human = group[human_column].dropna().tolist()
        if len(human) < 10:
            continue
        comet_r = None
        if "comet" in group.columns and not group["comet"].isna().all():
            comet_r, _ = _spearman(human, group["comet"].dropna().reindex(group.index).fillna(0).tolist())
        for kiwi in kiwi_cols:
            if group[kiwi].isna().all():
                continue
            kiwi_r, _ = _spearman(human, group[kiwi].dropna().reindex(group.index).fillna(0).tolist())
            rows.append({
                "lang": str(lang),
                "annotation_tier": str(ann_tier),
                "resource_tier": lang_to_tier.get(str(lang), "unknown"),
                "kiwi_variant": kiwi,
                "comet_spearman_r": comet_r,
                "kiwi_spearman_r": kiwi_r,
                "kiwi_advantage": (kiwi_r - comet_r) if comet_r is not None else None,
                "n": len(human),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["kiwi_variant", "annotation_tier", "lang"]).reset_index(drop=True)


def run_direction_analysis(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    resource_tiers: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Correlation analysis split by translation direction (X→en vs. en→X).

    Metrics like COMET and BERTScore were trained predominantly on en→X data.
    If they are less calibrated for X→en, we expect lower Spearman r for
    zh-en and he-en compared to en→X languages of similar resource level.

    Returns:
        DataFrame like run_correlation_analysis() with a 'direction' column.
    """
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS

    lang_to_tier = {lang: tier for tier, langs in resource_tiers.items() for lang in langs}
    lang_to_family = {lang: fam for fam, langs in LANGUAGE_FAMILIES.items() for lang in langs}
    lang_to_script = {lang: stype for stype, langs in SCRIPT_TYPES.items() for lang in langs}
    lang_to_dir = {lang: d for d, langs in TRANSLATION_DIRECTIONS.items() for lang in langs}

    rows = []
    for lang, group in scores_df.groupby("lang"):
        human = group[human_column].tolist()
        direction = lang_to_dir.get(str(lang), "en_to_x")
        for metric in metric_columns:
            if metric not in group.columns or group[metric].isna().all():
                continue
            result = correlate_metric_vs_human(human, group[metric].tolist(), metric, str(lang))
            result["resource_tier"] = lang_to_tier.get(str(lang), "unknown")
            result["language_family"] = lang_to_family.get(str(lang), "unknown")
            result["script_type"] = lang_to_script.get(str(lang), "unknown")
            result["direction"] = direction
            rows.append(result)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["direction", "metric", "lang"]).reset_index(drop=True)


def run_morphology_analysis(
    corr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-language correlations by morphological type.

    Tests whether metrics are more reliable for isolating languages (zh, th, lo —
    minimal inflection, overlap-based metrics benefit from predictable tokens) vs.
    agglutinative (tr, fi, ka — many suffixes → BLEU/ChrF hurt more) vs.
    fusional (de, ru — irregular inflection).

    Args:
        corr_df: Output of run_correlation_analysis(), must have a 'lang' column.

    Returns:
        Aggregated DataFrame with a 'morphology_type' column.
    """
    lang_to_morph = {lang: mtype for mtype, langs in MORPHOLOGY_TYPES.items() for lang in langs}
    df = corr_df.copy()
    df["morphology_type"] = df["lang"].map(lang_to_morph).fillna("fusional")
    cols = [c for c in _SUMMARY_COLS if c in df.columns]
    return (
        df.groupby(["metric", "morphology_type"])[cols]
        .mean()
        .reset_index()
        .sort_values(["metric", "morphology_type"])
    )


def run_length_analysis(
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
    resource_tiers: Optional[dict[str, list[str]]] = None,
    bins: tuple[int, int] = (10, 30),
) -> pd.DataFrame:
    """Correlation analysis stratified by source sentence length.

    Neural metrics rely on semantic context; for very short sentences (< 10 tokens)
    there is less signal. Hypothesis: COMET reliability drops more on short segments
    than surface metrics like BLEU (which suffers equally at all lengths).

    Args:
        bins: Token count boundaries (short < bins[0], long > bins[1]).

    Returns:
        DataFrame like run_correlation_analysis() with a 'length_bin' column.
    """
    if "source" not in scores_df.columns:
        return pd.DataFrame()
    if resource_tiers is None:
        resource_tiers = RESOURCE_TIERS

    lang_to_tier = {lang: tier for tier, langs in resource_tiers.items() for lang in langs}

    df = scores_df.copy()
    df["_src_len"] = df["source"].fillna("").str.split().str.len()

    def _bin(n):
        if n < bins[0]:
            return f"short (<{bins[0]})"
        elif n <= bins[1]:
            return f"medium ({bins[0]}–{bins[1]})"
        return f"long (>{bins[1]})"

    df["length_bin"] = df["_src_len"].apply(_bin)

    rows = []
    for (lang, length_bin), group in df.groupby(["lang", "length_bin"]):
        if len(group) < 20:
            continue
        human = group[human_column].tolist()
        for metric in metric_columns:
            if metric not in group.columns or group[metric].isna().all():
                continue
            result = correlate_metric_vs_human(human, group[metric].tolist(), metric, str(lang))
            result["resource_tier"] = lang_to_tier.get(str(lang), "unknown")
            result["length_bin"] = length_bin
            rows.append(result)

    if not rows:
        return pd.DataFrame()
    bin_order = [f"short (<{bins[0]})", f"medium ({bins[0]}–{bins[1]})", f"long (>{bins[1]})"]
    result_df = pd.DataFrame(rows)
    result_df["length_bin"] = pd.Categorical(result_df["length_bin"], categories=bin_order, ordered=True)
    return result_df.sort_values(["length_bin", "metric", "lang"]).reset_index(drop=True)


def run_rater_agreement_analysis(
    span_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    metric_columns: list[str],
    human_column: str = "quality_score",
) -> pd.DataFrame:
    """Correlate inter-rater agreement with metric-human correlation.

    For segments where professional MQM raters agree (all flag an error, or all
    flag no error), the human quality signal is cleaner, and metrics should
    correlate better. Segments with low agreement are ambiguous cases where even
    human judges disagree — not a fair test for any metric.

    Uses span_df (raw Google TSV data with a 'rater' column) to compute per-segment
    rater agreement, then bins segments into high/low agreement and computes
    per-bin metric-human Spearman r.

    Returns:
        DataFrame with 'agreement_bin' (high/low) column, or empty if rater
        data is unavailable.
    """
    if span_df.empty or "rater" not in span_df.columns or "segment_id" not in span_df.columns:
        return pd.DataFrame()
    if "segment_id" not in scores_df.columns:
        return pd.DataFrame()

    # Compute per-segment error fraction (fraction of raters who flagged any error)
    has_error = span_df["severity"].isin(["major", "minor", "critical"])
    rater_counts = span_df.groupby("segment_id")["rater"].nunique()
    error_counts = span_df[has_error].groupby("segment_id")["rater"].nunique()
    error_frac = (error_counts / rater_counts).fillna(0).rename("error_frac")

    # Agreement: 0 or 1 = all raters agree; 0.4–0.6 = maximum disagreement
    seg_agreement = error_frac.apply(lambda f: "high" if f <= 0.2 or f >= 0.8 else "low")
    seg_agreement.name = "agreement_bin"

    df = scores_df.merge(seg_agreement.reset_index(), on="segment_id", how="left")
    df["agreement_bin"] = df["agreement_bin"].fillna("high")  # single-rater segs default to high

    rows = []
    for (lang, agreement_bin), group in df.groupby(["lang", "agreement_bin"]):
        if len(group) < 20 or group[human_column].isna().all():
            continue
        human = group[human_column].tolist()
        for metric in metric_columns:
            if metric not in group.columns or group[metric].isna().all():
                continue
            sr, _ = _spearman(human, group[metric].fillna(0).tolist())
            rows.append({
                "lang": str(lang), "metric": metric,
                "agreement_bin": agreement_bin,
                "spearman_r": sr, "n": len(group),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["agreement_bin", "metric", "lang"]).reset_index(drop=True)
