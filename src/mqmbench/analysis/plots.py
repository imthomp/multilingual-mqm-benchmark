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


def plot_nontranslation_detection(scores_df: pd.DataFrame, out_path):
    """Bar chart of severe-error detection recall per metric (Tier 1a MQM only)."""
    mqm_df = scores_df[scores_df["annotation_tier"] == AnnotationTier.HUMAN_MQM].copy()
    if mqm_df.empty:
        return

    metric_cols = [c for c in ["bleu", "chrf", "bertscore", "comet", "xcomet", "gemba"]
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
