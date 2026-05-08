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
    compute_metric_correlations,
    kiwi_vs_comet_by_tier,
    reference_quality_effect,
    run_correlation_analysis,
    run_domain_analysis,
    run_system_level_analysis,
    run_williams_tests,
    summarize_by_family,
    summarize_by_script,
    summarize_by_tier,
)
from mqmbench.analysis.plots import (
    plot_correlation_by_language_family,
    plot_correlation_by_resource_level,
    plot_correlation_by_script_type,
    plot_domain_analysis,
    plot_kiwi_vs_comet_by_tier,
    plot_metric_category_heatmap,
    plot_metric_correlation_matrix,
    plot_nontranslation_detection,
    plot_reference_quality_effect,
    plot_score_distributions,
    plot_system_vs_segment_correlation,
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

    logger.info("=== Domain-stratified Analysis ===")
    domain_df = run_domain_analysis(full_scores_df, metric_columns,
                                    resource_tiers=resource_tiers)
    if not domain_df.empty:
        logger.info(f"  Domain analysis: {len(domain_df)} (lang×domain×metric) rows, "
                    f"domains: {sorted(domain_df['domain'].unique())}")
    else:
        logger.info("  Skipped (no 'domain' column in scores)")

    logger.info("=== System-level Analysis ===")
    sys_corr_df = run_system_level_analysis(full_scores_df, metric_columns,
                                            resource_tiers=resource_tiers)
    if not sys_corr_df.empty:
        logger.info(f"  System-level analysis: {len(sys_corr_df)} (lang×metric) rows")
    else:
        logger.info("  Skipped (no 'system' column in scores)")

    logger.info("=== Inter-metric Correlation Matrix ===")
    metric_corr_matrix = compute_metric_correlations(full_scores_df, metric_columns)
    if not metric_corr_matrix.empty:
        logger.info(f"  Metric correlation matrix: {metric_corr_matrix.shape}")

    logger.info("=== Reference Quality Effect ===")
    ref_quality_df = reference_quality_effect(full_scores_df, metric_columns,
                                              resource_tiers=resource_tiers)
    if not ref_quality_df.empty:
        logger.info(f"  Reference quality effect: {len(ref_quality_df)} rows")
    else:
        logger.info("  Skipped (missing COMET or Kiwi columns)")

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
    if not domain_df.empty:
        domain_df.to_csv(output_dir / "domain_analysis.csv", index=False)
    if not sys_corr_df.empty:
        sys_corr_df.to_csv(output_dir / "system_level_correlations.csv", index=False)
    if not metric_corr_matrix.empty:
        metric_corr_matrix.to_csv(output_dir / "metric_correlation_matrix.csv")
    if not ref_quality_df.empty:
        ref_quality_df.to_csv(output_dir / "reference_quality_effect.csv", index=False)

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
    if not domain_df.empty:
        plot_domain_analysis(domain_df, plots_dir / "domain_analysis.png")
    plot_score_distributions(full_scores_df, metric_columns,
                             resource_tiers=resource_tiers,
                             out_path=plots_dir / "score_distributions.png")
    if not metric_corr_matrix.empty:
        plot_metric_correlation_matrix(metric_corr_matrix,
                                       plots_dir / "metric_correlation_matrix.png")
    if not sys_corr_df.empty:
        plot_system_vs_segment_correlation(corr_df, sys_corr_df,
                                           plots_dir / "system_vs_segment.png")
    if not ref_quality_df.empty:
        plot_reference_quality_effect(ref_quality_df,
                                      plots_dir / "reference_quality_effect.png")

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
        "domain_analysis": domain_df,
        "system_level_correlations": sys_corr_df,
        "metric_correlation_matrix": metric_corr_matrix,
        "reference_quality_effect": ref_quality_df,
    }


def _run_metrics(scores_df: pd.DataFrame, cfg, checkpoint_dir: Path | None = None) -> list[str]:
    """Run each configured metric and add score columns to scores_df in place.

    If checkpoint_dir is provided:
    - Saves a checkpoint CSV after each metric completes.
    - On startup, reads any existing checkpoint and merges already-computed
      columns back into scores_df so those metrics are skipped this run.
      This lets a failed job resume from where it left off.
    """
    metrics_to_run = list(cfg.metrics.run)
    added_columns = []

    # Resume from checkpoint: merge any previously computed metric columns.
    # Uses segment_id merge so new rows (e.g. added Tier 2 languages) get NaN
    # for pre-computed metrics; those metrics are then recomputed from scratch.
    if checkpoint_dir is not None:
        ckpt_path = checkpoint_dir / "scores_checkpoint.csv"
        if ckpt_path.exists():
            try:
                ckpt = pd.read_csv(ckpt_path)
                ckpt_metric_cols = [c for c in ckpt.columns
                                    if c in metrics_to_run and c not in scores_df.columns]
                if ckpt_metric_cols:
                    if "segment_id" in ckpt.columns and "segment_id" in scores_df.columns:
                        merged = scores_df[["segment_id"]].merge(
                            ckpt[["segment_id"] + ckpt_metric_cols],
                            on="segment_id", how="left",
                        )
                        for col in ckpt_metric_cols:
                            scores_df[col] = merged[col].values
                    elif len(ckpt) == len(scores_df):
                        scores_df[ckpt_metric_cols] = ckpt[ckpt_metric_cols].values
                    else:
                        logger.warning(
                            f"Checkpoint row count ({len(ckpt)}) differs from current data "
                            f"({len(scores_df)}) and no segment_id — recomputing all metrics"
                        )
                        ckpt_metric_cols = []

                    # Only skip metrics where every segment has a valid score
                    skipped, recomputing = [], []
                    for col in ckpt_metric_cols:
                        if scores_df[col].notna().all():
                            skipped.append(col)
                            added_columns.append(col)
                        else:
                            missing_n = int(scores_df[col].isna().sum())
                            recomputing.append(f"{col}({missing_n} missing)")
                            del scores_df[col]
                    if skipped:
                        logger.info(f"Resuming from checkpoint — skipping: {skipped}")
                    if recomputing:
                        logger.info(f"Checkpoint incomplete for: {recomputing} — will recompute")
            except Exception as exc:
                logger.warning(f"Could not load checkpoint ({exc}) — recomputing all metrics")

    def _checkpoint(name: str) -> None:
        if checkpoint_dir is not None:
            path = checkpoint_dir / "scores_checkpoint.csv"
            id_cols = [c for c in ("segment_id", "lang", "annotation_tier") if c in scores_df.columns]
            scores_df[id_cols + added_columns].to_csv(path, index=False)
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

    if "xcometxxl" in metrics_to_run:
        logger.info("Computing xCOMET-XXL...")
        from mqmbench.metrics import comet
        sources, hyps, refs = _text_lists(scores_df)
        scores_df["xcometxxl"] = comet.score_xcomet(
            sources, hyps, refs,
            model_name=cfg.metrics.xcometxxl.model,
            batch_size=cfg.metrics.xcometxxl.batch_size,
            gpus=cfg.metrics.xcometxxl.gpus,
        )
        added_columns.append("xcometxxl")
        _checkpoint("xcometxxl")

    if "cometkiwi" in metrics_to_run:
        logger.info("Computing COMET-Kiwi 2022 (reference-free)...")
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

    if "cometkiwi23" in metrics_to_run:
        logger.info("Computing COMET-Kiwi 2023 XL (reference-free)...")
        from mqmbench.metrics import comet
        sources, hyps, _ = _text_lists(scores_df)
        scores_df["cometkiwi23"] = comet.score_kiwi(
            sources, hyps,
            model_name=cfg.metrics.cometkiwi23.model,
            batch_size=cfg.metrics.cometkiwi23.batch_size,
            gpus=cfg.metrics.cometkiwi23.gpus,
        )
        added_columns.append("cometkiwi23")
        _checkpoint("cometkiwi23")

    if getattr(cfg.metrics, "run_gemba", False):
        # GEMBA is an LLM inference metric — running it on 700k+ segments with a
        # local Llama model would take 30–100 hours. Restrict to MQM tier only
        # (~85k segments), which is also where it makes most analytical sense
        # (professional human judgments vs. LLM-as-judge comparison).
        gemba_df = scores_df[scores_df["annotation_tier"] == "human_mqm"].copy()
        logger.info(f"Computing GEMBA-MQM on MQM tier only ({len(gemba_df)} segments)...")
        from mqmbench.metrics import gemba
        g_src, g_hyp, g_ref = _text_lists(gemba_df)
        raw_penalties = gemba.score(g_src, g_hyp, g_ref,
                                    model_name=cfg.metrics.gemba_model)
        scores_df["gemba"] = float("nan")
        scores_df.loc[gemba_df.index, "gemba"] = [1.0 / (1.0 + p) for p in raw_penalties]
        added_columns.append("gemba")
        _checkpoint("gemba")

    # Ensemble: average of available neural metrics (no GPU cost)
    neural = [c for c in ["comet", "xcomet", "cometkiwi", "cometkiwi23"]
              if c in scores_df.columns]
    if len(neural) >= 2:
        scores_df["ensemble"] = scores_df[neural].mean(axis=1)
        added_columns.append("ensemble")
        logger.info(f"Ensemble metric computed from: {neural}")

    return added_columns


def _text_lists(df: pd.DataFrame) -> tuple[list, list, list]:
    """Extract source/hypothesis/reference as plain string lists (NaN → empty string)."""
    def _to_str(col):
        return df[col].fillna("").astype(str).tolist()
    return _to_str("source"), _to_str("hypothesis"), _to_str("reference")
