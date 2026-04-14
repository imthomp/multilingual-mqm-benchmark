# Time Log — Multilingual MQM Benchmark

| Date | Time Range | Hours | Description |
|-|-|-|-|
| 01-16-2025 | 8-10AM | 2 | Brainstorming project ideas |
| 02-21-2026 | 9-11PM | 2 | Refining ideas, writing proposal (README) |
| 02-22-2026 | 9-10AM | 1 | Project kickoff: scaffolding, data pipeline stubs, metric stubs |
| 02-23-2026 | 1-2PM | 1 | Downloading data from MQM |
| 02-23-2026 | 2-4PM | 2 | Parsing and search for references |
| 04-05-2026 | 5PM-12AM | 7 | Pivoted to public data (WMT MQM/DA + FLORES-200 + NLLB-200); rewrote data loaders, correlation module, plots, pipeline, SLURM scripts; added AnnotationTier constants and shared data utils; code review and simplification pass |
| 04-06-2026 | 9AM-2PM | 5 | Wired Dr. Fulda language-family/script-type analysis into pipeline (summarize_by_script, summarize_by_family, two new plots); installed pytest; confirmed 11/11 tests pass; smoke test validated Tier 1a (49k MQM) + Tier 1b (135k DA) — Tier 2 NLLB needs SLURM as expected |
| 04-06-2026 | 9-11PM | 2 | Diagnosed SLURM job 11262595 timeout (CUDA driver incompatibility on m9g → CPU fallback); fixed all SLURM scripts to use dw/matrix (A100); resubmitted synthetic MQM job (11304485, running on m13l); investigated Ukrainian near-zero correlation — root cause: domain "other" (Telegram messages, avg 12 words) vs. news for all other languages; added `domain` column to DA/MQM loaders and correlation output |
| 04-10-2026 | 8AM-12PM | 4 | CUDA incompatibility chain (torch cu130 → cu124 wheel fix); cuDNN/NCCL missing lib installs; setuptools pkg_resources pin; multiple SLURM partition/QoS iterations (m9g → dw/matrix → m13l); first successful full pipeline run (job 11463513, 39 min on m13l, BLEU/ChrF/BERTScore/COMET) |
| 04-12-2026 | 9AM-5PM | 5 | Analyzed full correlation results; investigated Ukrainian near-zero anomaly (domain "other"/Telegram, single annotator); added bootstrap CIs (Spearman 95%, 1000 resamples); fixed Tier 2 all-no_error bug (updated GEMBA-MQM few-shot prompt to force critical evaluation); deleted stale cache; queued chained synthetic-MQM → full-pipeline jobs (11465030 → 11465038) |
| | | **Total: 30** | |