"""Plotting functions for benchmark analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mqmbench.constants import AnnotationTier


def plot_correlation_by_resource_level(corr_df: pd.DataFrame, out_path):
    """Bar plot of Kendall's Tau by resource tier across all metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=corr_df, x="metric", y="kendall_tau", hue="resource_tier",
        hue_order=["high", "medium", "low"], errorbar=None,
        palette="viridis", ax=ax,
    )
    ax.set_title("Metric Correlation with Human Quality by Resource Tier")
    ax.set_ylabel("Kendall's Tau")
    ax.set_xlabel("Metric")
    ax.legend(title="Resource Tier")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation_by_script_type(corr_df: pd.DataFrame, out_path):
    """Bar plot of Kendall's Tau by script type (logographic vs phonographic).

    Addresses Dr. Fulda's suggestion to investigate pictographic vs. phonographic
    language differences. BERTScore in particular may struggle with logographic
    scripts where tokenization differs fundamentally from alphabetic languages.
    """
    if "script_type" not in corr_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    script_order = sorted(corr_df["script_type"].dropna().unique())
    sns.barplot(
        data=corr_df, x="metric", y="kendall_tau", hue="script_type",
        hue_order=script_order, errorbar=None,
        palette="Set2", ax=ax,
    )
    ax.set_title("Metric Correlation by Script Type\n"
                 "(logographic=character-based, alphabetic/abjad/abugida=phonographic)")
    ax.set_ylabel("Kendall's Tau")
    ax.set_xlabel("Metric")
    ax.legend(title="Script Type")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation_by_language_family(corr_df: pd.DataFrame, out_path):
    """Heatmap of Kendall's Tau by language family and metric."""
    if "language_family" not in corr_df.columns:
        return
    pivot = corr_df.pivot_table(
        index="language_family", columns="metric",
        values="kendall_tau", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(5, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                linewidths=0.5, ax=ax)
    ax.set_title("Kendall's Tau by Language Family and Metric")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Language Family")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_category_heatmap(category_corr_df: pd.DataFrame, out_path):
    """Heatmap of metric correlation by MQM error category (Accuracy vs. Fluency)."""
    pivot = category_corr_df.pivot_table(
        index="metric", columns="error_category",
        values="kendall_tau", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0, fmt=".3f", ax=ax)
    ax.set_title("Kendall's Tau Correlation by MQM Error Category (Tier 1a Only)")
    ax.set_ylabel("Metric")
    ax.set_xlabel("Error Category")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_kiwi_vs_comet_by_tier(kiwi_df: pd.DataFrame, out_path):
    """Scatter comparing COMET-Kiwi (reference-free) vs. COMET (reference-based).

    One point per language. Points above the diagonal mean Kiwi beats COMET.
    Languages are coloured by resource tier so we can see whether low-resource
    pairs benefit from dropping the (potentially noisy) reference.
    """
    if kiwi_df.empty or "spearman_r_comet" not in kiwi_df.columns:
        return

    tier_colors = {"high": "#2166ac", "medium": "#4dac26", "low": "#d01c8b"}
    tier_order = ["high", "medium", "low"]

    measure_pairs = [
        ("spearman_r_comet", "spearman_r_kiwi", "Spearman r"),
        ("spa_comet", "spa_kiwi", "Soft Pairwise Accuracy (SPA)"),
    ]
    measure_pairs = [(c, k, t) for c, k, t in measure_pairs if c in kiwi_df.columns and k in kiwi_df.columns]
    if not measure_pairs:
        return

    n_plots = len(measure_pairs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    for ax, (comet_col, kiwi_col, title) in zip(axes, measure_pairs):
        for tier in tier_order:
            sub = kiwi_df[kiwi_df["resource_tier"] == tier]
            if sub.empty:
                continue
            ax.scatter(sub[comet_col], sub[kiwi_col], label=tier.capitalize(),
                       color=tier_colors[tier], s=70, alpha=0.88, zorder=3)
            # Label each point with the language code
            for _, row in sub.iterrows():
                ax.annotate(row["lang"].upper(), (row[comet_col], row[kiwi_col]),
                            fontsize=7, ha="center", va="bottom", color=tier_colors[tier])

        all_vals = pd.concat([kiwi_df[comet_col], kiwi_df[kiwi_col]]).dropna()
        if all_vals.empty:
            continue
        lo, hi = all_vals.min() - 0.03, all_vals.max() + 0.03
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="Equal performance")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"COMET (reference-based) — {title}")
        ax.set_ylabel(f"COMET-Kiwi (reference-free) — {title}")
        ax.set_title(f"Kiwi vs. COMET\n{title}")
        ax.legend(title="Resource tier", frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Reference-free vs. Reference-based: Does Reference Quality Matter?",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_tier_anomaly(anomaly_df: pd.DataFrame, out_path):
    """Two-panel figure showing why medium-resource languages beat high-resource ones.

    Left: per-year COMET Spearman r for each high-resource language (shows that
    individual-year correlations are strong, but aggregating across years depresses them).
    Right: cross-year variance vs. mean correlation, coloured by tier.
    """
    if anomaly_df.empty or "year" not in anomaly_df.columns:
        return

    comet_df = anomaly_df[anomaly_df["metric"] == "comet"].copy()
    multi_year = comet_df[comet_df["year"] != "unknown"].copy()
    if multi_year.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: per-year lines for high-resource languages ---
    ax = axes[0]
    high_df = multi_year[multi_year["resource_tier"] == "high"]
    if not high_df.empty:
        for lang in sorted(high_df["lang"].unique()):
            sub = high_df[high_df["lang"] == lang].sort_values("year")
            if sub["year"].nunique() < 2:
                continue
            ax.plot(sub["year"].astype(str), sub["year_spearman_r"],
                    marker="o", label=lang.upper(), linewidth=1.8)
        ax.set_title("COMET Spearman r by WMT Year\n(high-resource languages only)")
        ax.set_xlabel("WMT Year")
        ax.set_ylabel("COMET Spearman r")
        ax.legend(frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Right panel: variance vs. mean, coloured by tier ---
    ax = axes[1]
    tier_colors = {"high": "#2166ac", "medium": "#4dac26", "low": "#d01c8b"}
    var_df = (
        multi_year.groupby(["lang", "resource_tier"])["year_spearman_r"]
        .agg(mean_r="mean", std_r="std")
        .reset_index()
        .dropna(subset=["std_r"])
    )
    for tier in ["high", "medium", "low"]:
        sub = var_df[var_df["resource_tier"] == tier]
        if sub.empty:
            continue
        ax.scatter(sub["mean_r"], sub["std_r"], label=f"{tier.capitalize()} resource",
                   color=tier_colors[tier], s=70, alpha=0.88)
        for _, row in sub.iterrows():
            ax.annotate(row["lang"].upper(), (row["mean_r"], row["std_r"]),
                        fontsize=7, ha="center", va="bottom", color=tier_colors[tier])
    ax.set_xlabel("Mean per-year COMET Spearman r")
    ax.set_ylabel("Std dev of per-year Spearman r (heterogeneity)")
    ax.set_title("Cross-year Variance Explains Medium > High Anomaly\n"
                 "(higher std = aggregate correlation is depressed)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Why Do Medium-resource Languages Outperform High-resource Ones?",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_nontranslation_detection(scores_df: pd.DataFrame, out_path):
    """Bar chart of severe-error detection recall per metric (Tier 1a MQM only)."""
    mqm_df = scores_df[scores_df["annotation_tier"] == AnnotationTier.HUMAN_MQM].copy()
    if mqm_df.empty:
        return

    metric_cols = [c for c in ["bleu", "chrf", "bertscore", "comet", "xcomet", "cometkiwi", "gemba"]
                   if c in mqm_df.columns]
    if not metric_cols:
        return

    q25 = mqm_df["quality_score"].quantile(0.25)
    severe_mask = mqm_df["quality_score"] <= q25
    if severe_mask.sum() == 0:
        return

    rates = {
        m: (severe_mask & (mqm_df[m] <= mqm_df[m].quantile(0.25))).sum() / severe_mask.sum()
        for m in metric_cols
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates.keys(), rates.values(), color=sns.color_palette("muted", len(rates)))
    ax.axhline(0.25, linestyle="--", color="gray", linewidth=1, label="Random baseline (25%)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Severe-error detection rate (recall@Q1)")
    ax.set_xlabel("Metric")
    ax.set_title("Non-translation / Severe-error Detection Rate per Metric\n"
                 "(Tier 1a MQM only — bottom-quartile human quality)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
