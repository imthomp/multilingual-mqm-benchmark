"""Generate presentation-quality figures from correlations.csv.

Run from the project root:
    python scripts/make_presentation_plots.py

Outputs to results/plots/presentation_*.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS = Path("results")
OUT = RESULTS / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METRIC_LABELS = {
    "bleu": "BLEU",
    "chrf": "ChrF",
    "bertscore": "BERTScore",
    "comet": "COMET",
    "xcomet": "xCOMET",
    "cometkiwi": "COMET-Kiwi",
}

TIER_COLORS = {
    "high":   "#2166ac",
    "medium": "#4dac26",
    "low":    "#d01c8b",
}

TIER_ORDER = ["high", "medium", "low"]


def load_corr():
    path = RESULTS / "correlations.csv"
    if not path.exists():
        sys.exit(f"correlations.csv not found at {path}")
    df = pd.read_csv(path)
    df["metric_label"] = df["metric"].map(lambda m: METRIC_LABELS.get(m, m.upper()))
    return df


# ---------------------------------------------------------------------------
# Figure 1: Spearman r by metric and resource tier (grouped bar)
# ---------------------------------------------------------------------------
def plot_tier_bar(df: pd.DataFrame):
    metrics = [m for m in ["bleu", "chrf", "bertscore", "comet", "xcomet", "cometkiwi"]
               if m in df["metric"].unique()]

    # Average across languages within tier, exclude NaN (Tier 2 pending)
    tier_df = (
        df[df["spearman_r"].notna()]
        .groupby(["metric", "resource_tier"])["spearman_r"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.25
    offsets = [-width, 0, width]

    for i, tier in enumerate(TIER_ORDER):
        vals = []
        for m in metrics:
            row = tier_df[(tier_df["metric"] == m) & (tier_df["resource_tier"] == tier)]
            vals.append(row["spearman_r"].values[0] if len(row) else np.nan)
        bars = ax.bar(x + offsets[i], vals, width, label=tier.capitalize(),
                      color=TIER_COLORS[tier], alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m.upper()) for m in metrics])
    ax.set_ylabel("Spearman r (avg. per resource tier)")
    ax.set_title("MT Metric Correlation with Human Judgment\nby Resource Tier")
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend(title="Resource tier", frameon=False)

    # Annotate surprise finding
    ax.annotate("Medium > High\n(domain variance effect)",
                xy=(x[-1] + offsets[1], tier_df[(tier_df["metric"] == metrics[-1]) &
                    (tier_df["resource_tier"] == "medium")]["spearman_r"].values[0] + 0.02),
                xytext=(x[-1] - 0.8, 0.75),
                fontsize=11, color="#4dac26",
                arrowprops=dict(arrowstyle="->", color="#4dac26", lw=1.2))

    plt.tight_layout()
    path = OUT / "presentation_tier_bar.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Per-language Spearman heatmap
# ---------------------------------------------------------------------------
def plot_language_heatmap(df: pd.DataFrame):
    metrics = [m for m in ["bleu", "chrf", "bertscore", "comet", "xcomet", "cometkiwi"]
               if m in df["metric"].unique()]

    pivot = df.pivot_table(
        index="lang", columns="metric", values="spearman_r", aggfunc="mean"
    )[metrics]

    # Sort languages by resource tier then spearman_r of best metric
    lang_meta = df[["lang", "resource_tier"]].drop_duplicates().set_index("lang")
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    pivot = pivot.copy()
    pivot["_tier"] = pivot.index.map(lambda l: tier_rank.get(lang_meta.loc[l, "resource_tier"] if l in lang_meta.index else "low", 2))
    pivot = pivot.sort_values(["_tier", metrics[-1]], ascending=[True, False]).drop(columns="_tier")

    col_labels = [METRIC_LABELS.get(m, m.upper()) for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-0.2, vmax=0.9,
        linewidths=0.4, linecolor="white",
        xticklabels=col_labels,
        ax=ax, cbar_kws={"label": "Spearman r", "shrink": 0.8},
        annot_kws={"size": 11},
    )
    ax.set_title("Metric–Human Correlation per Language", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Tier divider lines + labels
    langs = list(pivot.index)
    tier_boundaries = []
    prev_tier = lang_meta.loc[langs[0], "resource_tier"] if langs[0] in lang_meta.index else "unknown"
    for i, lang in enumerate(langs[1:], 1):
        t = lang_meta.loc[lang, "resource_tier"] if lang in lang_meta.index else "unknown"
        if t != prev_tier:
            tier_boundaries.append(i)
            prev_tier = t
    for b in tier_boundaries:
        ax.axhline(b, color="black", linewidth=1.5)

    # Annotate Ukrainian outlier — place text inside the axes to avoid
    # overlapping with the colorbar on the right.
    if "uk" in langs:
        uk_idx = langs.index("uk")
        ax.annotate("domain outlier\n(Telegram)",
                    xy=(0, uk_idx + 0.5),
                    xytext=(len(metrics) / 2, uk_idx - 0.6),
                    fontsize=9, color="#c0392b", va="center", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c0392b",
                              alpha=0.85))

    plt.tight_layout()
    path = OUT / "presentation_language_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Data pipeline diagram (text-based, no data needed)
# ---------------------------------------------------------------------------
def plot_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    tiers = [
        ("Tier 1a\nWMT MQM", 1.5, "#2166ac",
         "de · zh · ru · he · es\nProfessional MQM annotations\n(span-level errors)"),
        ("Tier 1b\nWMT DA", 5.0, "#4dac26",
         "cs · tr · uk · ha · km · ps\nDirect Assessment scores\n(sentence-level)"),
        ("Tier 2\nSynthetic MQM", 8.5, "#d01c8b",
         "sw · ht · lo\nFLORES+ → NLLB-200 → Llama-3.1-8B\n(GEMBA-MQM style)"),
    ]

    for label, x, color, detail in tiers:
        # Box
        rect = mpatches.FancyBboxPatch((x - 1.3, 1.6), 2.6, 1.8,
            boxstyle="round,pad=0.15", linewidth=2,
            edgecolor=color, facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(x, 2.85, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
        ax.text(x, 2.0, detail, ha="center", va="center",
                fontsize=9.5, color="#333333", linespacing=1.5)

    # Arrow down to metrics
    for x in [1.5, 5.0, 8.5]:
        ax.annotate("", xy=(x, 1.55), xytext=(x, 1.3),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    ax.text(5.0, 0.9, "BLEU · ChrF · BERTScore · COMET",
            ha="center", va="center", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#888"))
    ax.text(5.0, 0.35, "Spearman / Kendall / Pairwise Accuracy  ·  Bootstrap 95% CIs  ·  Williams test",
            ha="center", va="center", fontsize=10.5, color="#555555")

    ax.set_title("Three-Tier Annotation Design — 13 Languages, 9 Families, 4 Script Types",
                 fontsize=14, pad=8)

    plt.tight_layout()
    path = OUT / "presentation_pipeline.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_pipeline_diagram()   # no data needed
    df = load_corr()
    plot_tier_bar(df)
    plot_language_heatmap(df)
    print("Done. Files in results/plots/")
