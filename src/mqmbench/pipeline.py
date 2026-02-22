"""Main evaluation pipeline.

Steps:
    1. Load annotation data
    2. Convert span-level annotations → sentence-level quality scores
    3. Run configured MT metrics
    4. Compute correlations against human scores
    5. Save results
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from mqmbench.analysis.correlation import run_correlation_analysis, summarize_by_tier
from mqmbench.config import init_settings, settings
from mqmbench.data.converter import annotations_to_sentence_scores
from mqmbench.data.loader import load_all_annotations


def run_pipeline(settings_file: Optional[str] = None) -> dict:
    """Run the full benchmark pipeline.

    Args:
        settings_file: Path to settings.toml. If None, uses defaults.

    Returns:
        Dict with keys 'correlations' (DataFrame) and 'tier_summary' (DataFrame).
    """
    if settings_file:
        init_settings(settings_file)

    annotations_dir = Path(settings.data.annotations_dir)
    output_dir = Path(settings.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {annotations_dir}...")
    raw_df = load_all_annotations(annotations_dir)
    print(f"  Loaded {len(raw_df)} annotation rows across {raw_df['lang'].nunique()} languages.")

    print("Converting span-level annotations to sentence scores...")
    scores_df = annotations_to_sentence_scores(raw_df)
    print(f"  {len(scores_df)} segments.")

    metric_columns = _run_metrics(scores_df, settings)

    print("Running correlation analysis...")
    corr_df = run_correlation_analysis(scores_df, metric_columns)
    tier_df = summarize_by_tier(corr_df)

    corr_path = output_dir / "correlations.csv"
    tier_path = output_dir / "tier_summary.csv"
    corr_df.to_csv(corr_path, index=False)
    tier_df.to_csv(tier_path, index=False)
    print(f"Results saved to {output_dir}/")

    return {"correlations": corr_df, "tier_summary": tier_df}


def _run_metrics(scores_df: pd.DataFrame, cfg) -> list[str]:
    """Run each configured metric and add score columns to scores_df in place."""
    metrics_to_run = list(cfg.metrics.run)
    added_columns = []

    sources = scores_df["source"].tolist()
    hyps = scores_df["hypothesis"].tolist()
    refs = scores_df["reference"].tolist()
    langs = scores_df["lang"].tolist()

    if "bleu" in metrics_to_run:
        print("Computing BLEU...")
        from mqmbench.metrics import bleu
        scores_df["bleu"] = bleu.score(sources, hyps, refs)
        added_columns.append("bleu")

    if "chrf" in metrics_to_run:
        print("Computing ChrF++...")
        from mqmbench.metrics import chrf
        scores_df["chrf"] = chrf.score(sources, hyps, refs)
        added_columns.append("chrf")

    if "bertscore" in metrics_to_run:
        print("Computing BERTScore...")
        from mqmbench.metrics import bertscore
        model = getattr(cfg.metrics.bertscore, "model", "microsoft/mdeberta-v3-base")
        batch_size = getattr(cfg.metrics.bertscore, "batch_size", 32)
        scores_df["bertscore"] = bertscore.score(sources, hyps, refs, model_type=model, batch_size=batch_size)
        added_columns.append("bertscore")

    if "comet" in metrics_to_run:
        print("Computing COMET...")
        from mqmbench.metrics import comet
        model = cfg.metrics.comet.model
        batch_size = cfg.metrics.comet.batch_size
        gpus = cfg.metrics.comet.gpus
        scores_df["comet"] = comet.score(sources, hyps, refs, model_name=model, batch_size=batch_size, gpus=gpus)
        added_columns.append("comet")

    if "xcomet" in metrics_to_run:
        print("Computing xCOMET...")
        from mqmbench.metrics import comet
        model = cfg.metrics.xcomet.model
        batch_size = cfg.metrics.xcomet.batch_size
        gpus = cfg.metrics.xcomet.gpus
        scores_df["xcomet"] = comet.score_xcomet(sources, hyps, refs, model_name=model, batch_size=batch_size, gpus=gpus)
        added_columns.append("xcomet")

    if getattr(cfg.metrics, "run_gemba", False):
        print("Computing GEMBA-MQM...")
        from mqmbench.metrics import gemba
        model_name = cfg.metrics.gemba_model
        raw_penalties = gemba.score(sources, hyps, refs, model_name=model_name)
        # Convert to quality score (higher = better) for consistent correlation direction
        scores_df["gemba"] = [1.0 / (1.0 + p) for p in raw_penalties]
        added_columns.append("gemba")

    return added_columns
