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
    analyze_metric_disagreements,
    analyze_tier_anomaly,
    kiwi_vs_comet_by_tier,
    run_correlation_analysis,
    run_williams_tests,
    summarize_by_family,
    summarize_by_script,
    summarize_by_tier,
)
from mqmbench.analysis.plots import (
    plot_correlation_by_language_family,
    plot_correlation_by_resource_level,
    plot_correlation_by_script_type,
    plot_kiwi_vs_comet_by_tier,
    plot_metric_category_heatmap,
    plot_nontranslation_detection,
    plot_tier_anomaly,
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
    metric_columns = _run_metrics(full_scores_df, settings, checkpoint_dir=output_dir)

    logger.info("=== Correlation Analysis ===")
    resource_tiers = {k: list(v) for k, v in settings.languages.resource_tiers.items()}
    corr_df = run_correlation_analysis(full_scores_df, metric_columns,
                                       resource_tiers=resource_tiers)

    # Exclude GEMBA metric scores on Tier 2 (synthetic_mqm) data: the same model
    # (Llama-3.1-8B-Instruct) generates both the synthetic annotations and the GEMBA
    # metric scores, creating circular evaluation. GEMBA correlations are valid for
    # Tier 1a/1b only.
    if "gemba" in metric_columns:
        corr_df = corr_df[~(
            (corr_df["metric"] == "gemba") &
            (corr_df["annotation_tier"] == AnnotationTier.SYNTHETIC_MQM)
        )].copy()
        logger.info("GEMBA correlations on synthetic_mqm tier excluded (circular evaluation).")

    tier_summary = summarize_by_tier(corr_df)

    # Category breakdown: use raw WMT MQM span annotations if available
    # (google/wmt-mqm-human-evaluation TSVs, loaded separately from the HF dataset).
    # Falls back gracefully to empty if span data is not present.
    cat_corr_df = pd.DataFrame()
    span_dir = Path(getattr(settings.data, "wmt_mqm_span_dir", ""))
    if span_dir and span_dir.exists():
        try:
            from mqmbench.data.wmt_mqm import load_wmt_mqm_spans
            from mqmbench.analysis.correlation import run_category_correlation
            logger.info("=== Category Correlation (Accuracy vs. Fluency) ===")
            span_df = load_wmt_mqm_spans(span_dir, lang_pairs=list(settings.data.wmt_mqm_pairs))
            if not span_df.empty:
                # Attach metric scores to span rows via segment_id lookup
                metric_lookup = (
                    full_scores_df[["segment_id"] + [c for c in metric_columns if c in full_scores_df.columns]]
                    .drop_duplicates("segment_id")
                )
                span_df = span_df.merge(metric_lookup, on="segment_id", how="left")
                cat_corr_df = run_category_correlation(
                    span_df, annotations_to_sentence_scores, metric_columns
                )
                logger.info(f"  Category correlations: {len(cat_corr_df)} rows")
        except Exception as exc:
            logger.warning(f"Category correlation skipped: {exc}")

    script_summary = summarize_by_script(corr_df)
    family_summary = summarize_by_family(corr_df)

    logger.info("=== Williams Tests ===")
    # Compare each metric against COMET (best-performing reference) per language.
    # Use the reference metric that's actually in the results; fall back to first available.
    reference_metric = next(
        (m for m in ["comet", "xcomet", "cometkiwi"] if m in metric_columns),
        metric_columns[0] if metric_columns else None,
    )
    williams_df = pd.DataFrame()
    if reference_metric and len(metric_columns) > 1:
        williams_df = run_williams_tests(
            full_scores_df, metric_columns,
            reference_metric=reference_metric,
        )
        logger.info(f"  Williams tests complete ({len(williams_df)} comparisons)")

    logger.info("=== Error Analysis ===")
    disagreement_df = analyze_metric_disagreements(
        full_scores_df, metric_columns, top_n=50,
    )
    logger.info(f"  Disagreement analysis: {len(disagreement_df)} rows")

    logger.info("=== COMET-Kiwi vs. COMET Comparison ===")
    kiwi_df = kiwi_vs_comet_by_tier(corr_df)
    if not kiwi_df.empty:
        logger.info(f"  Kiwi vs. COMET: {len(kiwi_df)} languages compared")
    else:
        logger.info("  Skipped (cometkiwi not in results — add to metrics.run)")

    logger.info("=== Medium > High Anomaly Analysis ===")
    anomaly_df = analyze_tier_anomaly(full_scores_df, metric_columns,
                                      resource_tiers=resource_tiers)
    if not anomaly_df.empty:
        logger.info(f"  Anomaly analysis: {len(anomaly_df)} lang×metric×year rows")
    else:
        logger.info("  Skipped (no 'year' column in scores — check wmt_mqm loader)")

    corr_df.to_csv(output_dir / "correlations.csv", index=False)
    tier_summary.to_csv(output_dir / "tier_summary.csv", index=False)
    script_summary.to_csv(output_dir / "script_type_summary.csv", index=False)
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)
    if not cat_corr_df.empty:
        cat_corr_df.to_csv(output_dir / "category_correlations.csv", index=False)
    if not williams_df.empty:
        williams_df.to_csv(output_dir / "williams_tests.csv", index=False)
    if not disagreement_df.empty:
        disagreement_df.to_csv(output_dir / "metric_disagreements.csv", index=False)
    if not kiwi_df.empty:
        kiwi_df.to_csv(output_dir / "kiwi_vs_comet.csv", index=False)
    if not anomaly_df.empty:
        anomaly_df.to_csv(output_dir / "tier_anomaly.csv", index=False)

    logger.info("=== Generating Plots ===")
    plot_correlation_by_resource_level(corr_df, plots_dir / "correlation_by_tier.png")
    plot_correlation_by_script_type(corr_df, plots_dir / "correlation_by_script.png")
    plot_correlation_by_language_family(corr_df, plots_dir / "correlation_by_family.png")
    if not cat_corr_df.empty:
        plot_metric_category_heatmap(cat_corr_df, plots_dir / "category_heatmap.png")
    plot_nontranslation_detection(full_scores_df, plots_dir / "nontranslation_detection.png")
    if not kiwi_df.empty:
        plot_kiwi_vs_comet_by_tier(kiwi_df, plots_dir / "kiwi_vs_comet.png")
    if not anomaly_df.empty:
        plot_tier_anomaly(anomaly_df, plots_dir / "tier_anomaly.png")

    logger.info(f"Pipeline complete. Results saved to {output_dir}/")
    return {
        "correlations": corr_df,
        "tier_summary": tier_summary,
        "script_type_summary": script_summary,
        "family_summary": family_summary,
        "category_correlations": cat_corr_df,
        "williams_tests": williams_df,
        "metric_disagreements": disagreement_df,
        "kiwi_vs_comet": kiwi_df,
        "tier_anomaly": anomaly_df,
    }


def _run_metrics(scores_df: pd.DataFrame, cfg, checkpoint_dir: Path | None = None) -> list[str]:
    """Run each configured metric and add score columns to scores_df in place.

    If checkpoint_dir is provided, saves scores_df to a checkpoint CSV after
    each metric completes so partial results survive a job timeout.
    """
    metrics_to_run = list(cfg.metrics.run)
    added_columns = []

    def _checkpoint(name: str) -> None:
        if checkpoint_dir is not None:
            path = checkpoint_dir / "scores_checkpoint.csv"
            scores_df[["lang", "annotation_tier"] + added_columns].to_csv(path, index=False)
            logger.info(f"Checkpoint saved after {name}: {path}")

    if "bleu" in metrics_to_run:
        logger.info("Computing BLEU...")
        from mqmbench.metrics import bleu
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["bleu"] = bleu.score(sources, hyps, refs)
        added_columns.append("bleu")
        _checkpoint("bleu")

    if "chrf" in metrics_to_run:
        logger.info("Computing ChrF++...")
        from mqmbench.metrics import chrf
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["chrf"] = chrf.score(sources, hyps, refs)
        added_columns.append("chrf")
        _checkpoint("chrf")

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
        _checkpoint("bertscore")

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
        _checkpoint("comet")

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
        _checkpoint("xcomet")

    if "cometkiwi" in metrics_to_run:
        logger.info("Computing COMET-Kiwi (reference-free)...")
        from mqmbench.metrics import comet
        sources, hyps, _ = _text_lists(scores_df)
        scores_df["cometkiwi"] = comet.score_kiwi(
            sources, hyps,
            model_name=cfg.metrics.cometkiwi.model,
            batch_size=cfg.metrics.cometkiwi.batch_size,
            gpus=cfg.metrics.cometkiwi.gpus,
        )
        added_columns.append("cometkiwi")
        _checkpoint("cometkiwi")

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
