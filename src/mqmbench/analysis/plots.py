"""Plotting functions for benchmark analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mqmbench.constants import AnnotationTier


def plot_correlation_by_resource_level(corr_df: pd.DataFrame, out_path):
    """Bar plot of Spearman r by resource tier, with SD error bars and per-language points.

    Error bars show ±1 SD across languages in each tier (spread = how consistent
    the metric is within the tier). Individual language points are overlaid so
    reviewers can see the full distribution, not just the average.
    """
    fig, ax = plt.subplots(figsize=(max(10, corr_df["metric"].nunique() * 1.4), 6))
    hue_order = [t for t in ["high", "medium", "low"] if t in corr_df["resource_tier"].values]
    sns.barplot(
        data=corr_df, x="metric", y="spearman_r", hue="resource_tier",
        hue_order=hue_order, errorbar="sd",
        palette="viridis", ax=ax, alpha=0.75,
    )
    sns.stripplot(
        data=corr_df, x="metric", y="spearman_r", hue="resource_tier",
        hue_order=hue_order, palette="viridis", ax=ax,
        dodge=True, size=4, alpha=0.55, legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_title("Spearman ρ vs. Human Quality by Resource Tier\n"
                 "(bars = tier mean ± 1 SD; points = individual languages)")
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("Metric")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(hue_order)], [l.capitalize() for l in labels[:len(hue_order)]],
              title="Resource Tier", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_correlation_by_script_type(corr_df: pd.DataFrame, out_path):
    """Bar plot of Spearman r by script type with SD error bars and individual points.

    Addresses Dr. Fulda's suggestion to investigate pictographic vs. phonographic
    language differences. BERTScore in particular may struggle with logographic
    scripts where tokenization differs fundamentally from alphabetic languages.
    """
    if "script_type" not in corr_df.columns:
        return
    fig, ax = plt.subplots(figsize=(max(10, corr_df["metric"].nunique() * 1.4), 6))
    script_order = sorted(corr_df["script_type"].dropna().unique())
    sns.barplot(
        data=corr_df, x="metric", y="spearman_r", hue="script_type",
        hue_order=script_order, errorbar="sd",
        palette="Set2", ax=ax, alpha=0.75,
    )
    sns.stripplot(
        data=corr_df, x="metric", y="spearman_r", hue="script_type",
        hue_order=script_order, palette="Set2", ax=ax,
        dodge=True, size=4, alpha=0.55, legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_title("Spearman ρ by Script Type\n"
                 "(bars = mean ± 1 SD across languages; points = individual languages)")
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("Metric")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(script_order)], labels[:len(script_order)],
              title="Script Type", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
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
    """Scatter comparing COMET-Kiwi variants (reference-free) vs. COMET (reference-based).

    One point per language per kiwi variant. Points above the diagonal mean Kiwi
    beats COMET. 95% bootstrap CI error bars are shown when available.
    Languages are coloured by resource tier.
    """
    if kiwi_df.empty or "spearman_r_comet" not in kiwi_df.columns:
        return

    tier_colors = {"high": "#2166ac", "medium": "#4dac26", "low": "#d01c8b"}
    tier_order = ["high", "medium", "low"]

    # Find available kiwi variants and measures
    kiwi_suffixes = [s for s in ["kiwi22", "kiwi23"] if f"spearman_r_{s}" in kiwi_df.columns]
    if not kiwi_suffixes:
        kiwi_suffixes = ["kiwi"] if "spearman_r_kiwi" in kiwi_df.columns else []
    if not kiwi_suffixes:
        return

    measure_pairs = [("spearman_r", "Spearman ρ"), ("spa", "SPA")]
    measure_pairs = [(m, t) for m, t in measure_pairs if f"{m}_comet" in kiwi_df.columns]

    n_cols = len(measure_pairs)
    n_rows = len(kiwi_suffixes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.5 * n_rows),
                             squeeze=False)

    labels = {"kiwi22": "Kiwi 2022", "kiwi23": "Kiwi 2023 XL", "kiwi": "COMET-Kiwi"}

    for row_i, suffix in enumerate(kiwi_suffixes):
        for col_i, (measure, title) in enumerate(measure_pairs):
            ax = axes[row_i][col_i]
            comet_col = f"{measure}_comet"
            kiwi_col = f"{measure}_{suffix}"
            ci_lo_comet = f"spearman_ci_lo_comet"
            ci_hi_comet = f"spearman_ci_hi_comet"
            ci_lo_kiwi = f"spearman_ci_lo_{suffix}"
            ci_hi_kiwi = f"spearman_ci_hi_{suffix}"

            for tier in tier_order:
                sub = kiwi_df[kiwi_df["resource_tier"] == tier].dropna(subset=[comet_col, kiwi_col])
                if sub.empty:
                    continue
                xerr = yerr = None
                if measure == "spearman_r":
                    if ci_lo_comet in sub.columns and ci_hi_comet in sub.columns:
                        xerr = [sub[comet_col] - sub[ci_lo_comet],
                                sub[ci_hi_comet] - sub[comet_col]]
                    if ci_lo_kiwi in sub.columns and ci_hi_kiwi in sub.columns:
                        yerr = [sub[kiwi_col] - sub[ci_lo_kiwi],
                                sub[ci_hi_kiwi] - sub[kiwi_col]]
                ax.errorbar(sub[comet_col], sub[kiwi_col],
                            xerr=xerr, yerr=yerr,
                            fmt="o", color=tier_colors[tier], label=tier.capitalize(),
                            markersize=7, alpha=0.85, capsize=3, linewidth=0.8, zorder=3)
                for _, r in sub.iterrows():
                    ax.annotate(r["lang"].upper(), (r[comet_col], r[kiwi_col]),
                                fontsize=6.5, ha="center", va="bottom",
                                color=tier_colors[tier])

            all_vals = pd.concat([kiwi_df[comet_col], kiwi_df[kiwi_col]]).dropna()
            if all_vals.empty:
                continue
            lo, hi = all_vals.min() - 0.04, all_vals.max() + 0.04
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.45)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(f"COMET (reference-based) — {title}")
            ax.set_ylabel(f"{labels[suffix]} (reference-free) — {title}")
            ax.set_title(f"{labels[suffix]} vs. COMET | {title}\n"
                         "(error bars = 95% bootstrap CI; above diagonal = Kiwi wins)")
            if col_i == 0:
                ax.legend(title="Resource tier", frameon=False, fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.suptitle("Reference-free vs. Reference-based: Does Dropping the Reference Help?",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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


def plot_domain_analysis(domain_df: pd.DataFrame, out_path):
    """Grouped bar chart of Spearman ρ by domain (news vs. other/conversational).

    Shows that metric reliability is stable within domains but drops when domains
    are pooled — turning the Ukrainian domain confound into a methodological insight.
    """
    if domain_df.empty or "domain" not in domain_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    measure_col_title = [
        ("spearman_r", "Spearman ρ"),
        ("spa", "Soft Pairwise Accuracy"),
    ]

    for ax, (col, title) in zip(axes, measure_col_title):
        if col not in domain_df.columns:
            continue
        domain_order = sorted(domain_df["domain"].dropna().unique())
        palette = sns.color_palette("Set1", len(domain_order))
        sns.barplot(
            data=domain_df, x="metric", y=col, hue="domain",
            hue_order=domain_order, errorbar="sd",
            palette=palette, ax=ax, alpha=0.82,
        )
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.35)
        ax.set_title(f"Per-domain {title}\n(bars = mean ± 1 SD across languages in domain)")
        ax.set_ylabel(title)
        ax.set_xlabel("Metric")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Domain", frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Metric Reliability by Domain: News vs. Conversational",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_nontranslation_detection(scores_df: pd.DataFrame, out_path):
    """Bar chart of severe-error detection recall per metric (Tier 1a MQM only)."""
    mqm_df = scores_df[scores_df["annotation_tier"] == AnnotationTier.HUMAN_MQM].copy()
    if mqm_df.empty:
        return

    all_metrics = ["bleu", "chrf", "bertscore", "comet", "xcomet", "xcometxxl",
                   "cometkiwi", "cometkiwi23", "gemba"]
    metric_cols = [c for c in all_metrics if c in mqm_df.columns]
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
