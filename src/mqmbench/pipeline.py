"""Main evaluation pipeline for public datasets.

Steps:
    1. Load Tier 1a (WMT MQM), Tier 1b (WMT DA), Tier 2 (FLORES + NLLB + synthetic MQM)
    2. Convert span-level annotations to sentence scores where applicable
    3. Run configured MT metrics across all tiers
    4. Compute correlations against human/synthetic quality scores
    5. Generate plots and save results
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from mqmbench.analysis.correlation import (
    run_correlation_analysis,
    summarize_by_family,
    summarize_by_script,
    summarize_by_tier,
)
from mqmbench.analysis.plots import (
    plot_correlation_by_language_family,
    plot_correlation_by_resource_level,
    plot_correlation_by_script_type,
    plot_metric_category_heatmap,
    plot_nontranslation_detection,
)
from mqmbench.config import init_settings, settings
from mqmbench.constants import AnnotationTier
from mqmbench.data.converter import annotations_to_sentence_scores  # used for Tier 2 synthetic spans
from mqmbench.data.flores import load_flores
from mqmbench.data.nllb import generate_hypotheses
from mqmbench.data.synthetic_mqm import generate_synthetic_mqm
from mqmbench.data.wmt_da import load_wmt_da
from mqmbench.data.wmt_mqm import load_wmt_mqm

logger = logging.getLogger(__name__)


def run_pipeline(settings_file: Optional[str] = None) -> dict:
    """Run the full benchmark pipeline."""
    if settings_file:
        init_settings(settings_file)

    output_dir = Path(settings.data.results_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = settings.data.cache_dir
    all_scores = []

    logger.info("=== Tier 1a: WMT MQM ===")
    # RicardoRei/wmt-mqm-human-evaluation provides sentence-level scores directly;
    # no span conversion needed. Category analysis is not available from this source.
    mqm_scores_df = load_wmt_mqm(lang_pairs=list(settings.data.wmt_mqm_pairs))
    if not mqm_scores_df.empty:
        all_scores.append(mqm_scores_df)
        logger.info(f"  {len(mqm_scores_df)} segments, languages: {sorted(mqm_scores_df['lang'].unique())}")

    logger.info("=== Tier 1b: WMT DA ===")
    da_scores_df = load_wmt_da(target_langs=list(settings.data.wmt_da_target_langs))
    if not da_scores_df.empty:
        all_scores.append(da_scores_df)
        logger.info(f"  {len(da_scores_df)} segments, languages: {sorted(da_scores_df['lang'].unique())}")

    logger.info("=== Tier 2: Synthetic MQM ===")
    tier2_langs = list(getattr(settings.data, "tier2_langs", []))
    flores_df = load_flores(lang_codes=tier2_langs) if tier2_langs else pd.DataFrame()
    if not flores_df.empty:
        flores_hyp_df = generate_hypotheses(
            flores_df, model_name=settings.data.nllb_model, cache_dir=cache_dir,
        )
        synth_model = settings.metrics.gemba_model
        if not synth_model:
            logger.warning("metrics.gemba_model not set — skipping Tier 2 synthetic MQM.")
        else:
            synth_raw_df = generate_synthetic_mqm(
                flores_hyp_df, model_name=synth_model, cache_dir=cache_dir,
            )
            synth_scores_df = annotations_to_sentence_scores(synth_raw_df)
            synth_scores_df["annotation_tier"] = AnnotationTier.SYNTHETIC_MQM
            all_scores.append(synth_scores_df)
            logger.info(f"  {len(synth_scores_df)} segments, languages: {sorted(synth_scores_df['lang'].unique())}")

    if not all_scores:
        raise RuntimeError("No data loaded from any tier. Check settings and network access.")

    logger.info("=== Running Metrics ===")
    full_scores_df = pd.concat(all_scores, ignore_index=True)
    metric_columns = _run_metrics(full_scores_df, settings)

    logger.info("=== Correlation Analysis ===")
    resource_tiers = {k: list(v) for k, v in settings.languages.resource_tiers.items()}
    corr_df = run_correlation_analysis(full_scores_df, metric_columns,
                                       resource_tiers=resource_tiers)
    tier_summary = summarize_by_tier(corr_df)

    # Category breakdown requires raw span annotations; not available from
    # RicardoRei/wmt-mqm-human-evaluation (sentence-level only).
    cat_corr_df = pd.DataFrame()

    script_summary = summarize_by_script(corr_df)
    family_summary = summarize_by_family(corr_df)

    corr_df.to_csv(output_dir / "correlations.csv", index=False)
    tier_summary.to_csv(output_dir / "tier_summary.csv", index=False)
    script_summary.to_csv(output_dir / "script_type_summary.csv", index=False)
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)
    if not cat_corr_df.empty:
        cat_corr_df.to_csv(output_dir / "category_correlations.csv", index=False)

    logger.info("=== Generating Plots ===")
    plot_correlation_by_resource_level(corr_df, plots_dir / "correlation_by_tier.png")
    plot_correlation_by_script_type(corr_df, plots_dir / "correlation_by_script.png")
    plot_correlation_by_language_family(corr_df, plots_dir / "correlation_by_family.png")
    if not cat_corr_df.empty:
        plot_metric_category_heatmap(cat_corr_df, plots_dir / "category_heatmap.png")
    plot_nontranslation_detection(full_scores_df, plots_dir / "nontranslation_detection.png")

    logger.info(f"Pipeline complete. Results saved to {output_dir}/")
    return {
        "correlations": corr_df,
        "tier_summary": tier_summary,
        "script_type_summary": script_summary,
        "family_summary": family_summary,
        "category_correlations": cat_corr_df,
    }


def _run_metrics(scores_df: pd.DataFrame, cfg) -> list[str]:
    """Run each configured metric and add score columns to scores_df in place."""
    metrics_to_run = list(cfg.metrics.run)
    added_columns = []

    if "bleu" in metrics_to_run:
        logger.info("Computing BLEU...")
        from mqmbench.metrics import bleu
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["bleu"] = bleu.score(sources, hyps, refs)
        added_columns.append("bleu")

    if "chrf" in metrics_to_run:
        logger.info("Computing ChrF++...")
        from mqmbench.metrics import chrf
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["chrf"] = chrf.score(sources, hyps, refs)
        added_columns.append("chrf")

    if "bertscore" in metrics_to_run:
        logger.info("Computing BERTScore...")
        from mqmbench.metrics import bertscore
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["bertscore"] = bertscore.score(
            sources, hyps, refs,
            model_type=cfg.metrics.bertscore.model,
            batch_size=cfg.metrics.bertscore.batch_size,
        )
        added_columns.append("bertscore")

    if "comet" in metrics_to_run:
        logger.info("Computing COMET...")
        from mqmbench.metrics import comet
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["comet"] = comet.score(
            sources, hyps, refs,
            model_name=cfg.metrics.comet.model,
            batch_size=cfg.metrics.comet.batch_size,
            gpus=cfg.metrics.comet.gpus,
        )
        added_columns.append("comet")

    if "xcomet" in metrics_to_run:
        logger.info("Computing xCOMET...")
        from mqmbench.metrics import comet
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["xcomet"] = comet.score_xcomet(
            sources, hyps, refs,
            model_name=cfg.metrics.xcomet.model,
            batch_size=cfg.metrics.xcomet.batch_size,
            gpus=cfg.metrics.xcomet.gpus,
        )
        added_columns.append("xcomet")

    if getattr(cfg.metrics, "run_gemba", False):
        logger.info("Computing GEMBA-MQM...")
        from mqmbench.metrics import gemba
        sources, hyps, refs = _text_lists(scores_df)
        raw_penalties = gemba.score(sources, hyps, refs,
                                    model_name=cfg.metrics.gemba_model)
        scores_df["gemba"] = [1.0 / (1.0 + p) for p in raw_penalties]
        added_columns.append("gemba")

    return added_columns


def _text_lists(df: pd.DataFrame) -> tuple[list, list, list]:
    """Extract source/hypothesis/reference as plain string lists (NaN → empty string)."""
    def _to_str(col):
        return df[col].fillna("").astype(str).tolist()
    return _to_str("source"), _to_str("hypothesis"), _to_str("reference")
