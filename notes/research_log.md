# Research Log — Multilingual MQM Benchmark

## 2026-04-12 (Session 5, 9AM–5PM+)

### What we did

**CUDA debugging resolved** (carried over from Apr 10):
- Root cause: `torch 2.11.0+cu130` requires CUDA 13.0; all cluster nodes max at CUDA 12.8
- Fix: pinned `torch` to cu124 wheel via `pyproject.toml` uv index source; added `setuptools==69.5.1` to core deps to fix `pkg_resources` breakage from `setuptools>=72`
- cuDNN/NCCL missing after cu124 install → fixed with `uv pip install nvidia-cudnn-cu12 nvidia-nccl-cu12`
- First successful full pipeline run: job 11463513 on m13l, 39 minutes, BLEU/ChrF/BERTScore/COMET results for all tiers

**Bootstrap confidence intervals**:
- Added `_bootstrap_ci()` (percentile method, 1000 resamples, seed=42) to `correlation.py`
- `correlate_metric_vs_human()` now computes `spearman_ci_lo` / `spearman_ci_hi`; columns added to `run_correlation_analysis()` output and `correlations.csv`

**Tier 2 all-no_error fix**:
- Old GEMBA-MQM prompt caused Llama-3.1-8B to rate 100% of sw/ht/lo translations as NO ERRORS → quality_score=1.0 for all segments → NaN correlation (zero variance)
- New prompt: few-shot examples with actual errors force more critical evaluation
- Deleted stale cache files; queued chained jobs: 11465030 (synth re-annotation) → 11465038 (full pipeline)

**xCOMET added**:
- `settings.toml` `run` list now includes `"xcomet"`
- `run_comet.sh` time limit bumped to 8h; xCOMET-XL will run as part of job 11465038

**GEMBA circular evaluation fix**:
- Added filter in `pipeline.py`: GEMBA metric correlations on `synthetic_mqm` tier are excluded from `correlations.csv` (same model annotates and evaluates → circular)
- Logged with explanation; results for Tier 1a/1b unaffected

**Span-level category correlation (accuracy vs. fluency)**:
- Added `load_wmt_mqm_spans()` to `wmt_mqm.py` — reads Google's raw WMT MQM TSV files
- `pipeline.py` checks `settings.data.wmt_mqm_span_dir`; if set and directory exists, loads spans and runs `run_category_correlation()`
- Falls back gracefully if TSV dir is not present
- To activate: download TSVs from github.com/google/wmt-mqm-human-evaluation, set `wmt_mqm_span_dir = "data/wmt_mqm_spans"` in settings.toml

**Late session additions (9PM+)**:
- Added COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`) as reference-free metric — addresses key gap flagged in literature (reference quality unreliable for low-resource languages); model downloaded, wired into pipeline and settings
- Added pairwise accuracy to `correlate_metric_vs_human()` — WMT 2023-2024 primary segment-level meta-evaluation measure; included in `correlations.csv` and all summary tables
- Added Williams (1959) test (`williams_test()`, `run_williams_tests()`) — significance test for difference between two correlated correlations sharing a human criterion; outputs `williams_tests.csv` comparing each metric vs. COMET per language
- Added error analysis (`analyze_metric_disagreements()`) — identifies top-50 worst-disagreement segments per (lang, metric) by rank residual; outputs `metric_disagreements.csv` with source text, scores, and over/under-scored direction
- Jobs resubmitted on `m13h` (H200, idle) with `--qos=normal` after `cs/matrix` was blocked and estimated start was 6:45AM tomorrow; synth job running as of ~9:40PM
- Cloned `google/wmt-mqm-human-evaluation` TSV span data; loader working (641k span rows for de/zh/es/he); Russian TSV malformed upstream (trailing tab shifts JSON into severity column — not our bug)

### Results from job 11463513 (pre-Tier-2 fix)
- COMET dominated all tiers (Spearman 0.6–0.8 for high/medium resource)
- ChrF consistently outperformed BLEU
- Ukrainian: near-zero across ALL metrics (confirmed domain confound — Telegram/conversational, not news)
- Tier 2 (sw/ht/lo): all NaN — fixed with prompt update, awaiting rerun

### Jobs queued
- **11465030** (`mqmbench_synth`, dw): re-annotate sw/ht/lo with fixed prompt
- **11465038** (`mqmbench_comet`, dw, `afterok:11465030`): full pipeline with xCOMET

### Publishability status
- Core results (Tier 1a/1b, 4 metrics, 11 languages) are solid and reportable
- Bootstrap CIs now support statistical claims
- Tier 2 fix pending (jobs queued)
- Category correlation (accuracy vs. fluency): code complete, needs WMT span TSV download
- Ukrainian domain confound should be called out explicitly in the paper

### Future work (to strengthen toward publication)

**Short-term (would improve WMT/LREC-COLING submission):**
- MetricX-23 and BLEURT-20 — reviewers expect these; requires separate library installs and model downloads
- Soft Pairwise Accuracy (SPA) — stricter WMT 2024 variant of pairwise accuracy with tie calibration; pairwise accuracy is close but SPA is the exact official measure
- Validate Tier 2 synthetic annotations: even 200 human judgments on sw/ht/lo would transform "synthetic" into "validated synthetic"
- Accuracy vs. fluency category correlation — code complete; needs raw WMT span TSVs already downloaded; just set `wmt_mqm_span_dir` and re-run
- Russian span data — the 2022 enru TSV is malformed (trailing tab); could fetch the 2021 or 2023 version manually

**Medium-term (EMNLP Findings territory):**
- Reference-free vs. reference-based comparison by resource tier — is COMET-Kiwi competitive with COMET when references are low-quality? This is the practical payoff of adding Kiwi.
- Adapt a metric for low-resource: fine-tune COMET on synthetic MQM annotations for sw/ht/lo and test if it improves — would add the constructive element reviewers want
- Expand language coverage: Thai (abugida, no word boundaries), Myanmar (abugida), Amharic (Ge'ez script) — would strengthen the script-type story
- Quantify medium > high anomaly: test whether it disappears when controlling for number of WMT years and MT systems — would turn a finding into a methodological insight

**Framing note:** strongest submission angle is "what drives MT metric reliability across typologically diverse languages?" — diagnostic study + recommendation matrix, not pure negative result. COMET-Kiwi vs. COMET by resource tier is the key table to build that recommendation.

**Language expansion — zero new data collection needed:**

*From WMT DA (`RicardoRei/wmt-da-human-evaluation`) — already loaded, just add to `wmt_da_target_langs`:*
| Code | Language | Family | Script | Why interesting |
|------|----------|--------|--------|-----------------|
| bn | Bengali | Indo-European | Abugida (Bengali) | Large language, underrepresented in meta-eval |
| gu | Gujarati | Indo-European | Abugida (Gujarati) | Brahmic script variant |
| hi | Hindi | Indo-European | Abugida (Devanagari) | Huge speaker population, shared script with many languages |
| ta | Tamil | Dravidian | Abugida (Tamil) | Non-Indo-European, distinct script |
| ja | Japanese | Japonic | Logographic+syllabic | Mixed script (kanji+kana), no spaces |
| kk | Kazakh | Turkic | Cyrillic | Expands Turkic family beyond tr |
| xh | Xhosa | Niger-Congo | Latin | Click consonants, Bantu |
| zu | Zulu | Niger-Congo | Latin | Click consonants, Bantu, closely related to xh |
| et | Estonian | Uralic | Latin | Agglutinative, non-Indo-European |
| fi | Finnish | Uralic | Latin | Agglutinative, highly inflected |
| is | Icelandic | Indo-European | Latin | Low-resource, heavily inflected Germanic |
| lt | Lithuanian | Indo-European | Latin | Most archaic living IE language |
| lv | Latvian | Indo-European | Latin | Baltic, sibling of Lithuanian |
| pl | Polish | Indo-European | Latin | Slavic with complex morphology |
| fr | French | Indo-European | Latin | High-resource, useful sanity check |

*From FLORES+ + NLLB-200 (Tier 2, synthetic only — just add language codes to `tier2_langs`):*
| Code | Language | Family | Script | Why interesting |
|------|----------|--------|--------|-----------------|
| tha_Thai | Thai | Tai-Kadai | Thai | No word boundaries, same family as lo |
| mya_Mymr | Burmese | Sino-Tibetan | Myanmar (abugida) | Unique script, very low resource |
| amh_Ethi | Amharic | Afro-Asiatic | Ge'ez (unique) | Only Ge'ez script language in dataset |
| yor_Latn | Yoruba | Niger-Congo | Latin | Tonal, W. African, low resource |
| ibo_Latn | Igbo | Niger-Congo | Latin | Tonal, W. African, low resource |
| kat_Geor | Georgian | Language isolate | Georgian (unique) | Script isolate, no known relatives |
| hye_Armn | Armenian | Indo-European | Armenian (unique) | Unique script, IE outlier |
| uig_Arab | Uyghur | Turkic | Arabic (RTL) | Turkic in Arabic script — tests script vs. family |
| tel_Telu | Telugu | Dravidian | Abugida (Telugu) | Expands Dravidian beyond Tamil |
| kan_Knda | Kannada | Dravidian | Abugida (Kannada) | Expands Dravidian coverage |
| eus_Latn | Basque | Language isolate | Latin | Only language isolate with Latin script in the set |
| bod_Tibt | Tibetan | Sino-Tibetan | Tibetan (abugida) | Brahmic-derived, very low resource |

---

## 2026-04-06 (Session 3, 9–11PM)

### What we did
- **Diagnosed SLURM timeout**: Job 11262595 ran 4 hours on CPU — CUDA driver on m9g nodes (version 12080) incompatible with `torch 2.11.0+cu130`. PyTorch silently fell back to CPU; NLLB on CPU is ~2h/language.
- **Fixed SLURM scripts**: All three (`run_nllb.sh`, `run_synthetic_mqm.sh`, `run_comet.sh`) now use `--partition=dw --qos=matrix` (A100). `m13l` partition has the same driver issue.
- **Resubmitted**: Job 11304485 on `m13l` (sw + ht already cached from timeout run; only `lo` ~1012 segments remains, ~2h to complete).
- **Root-caused Ukrainian near-zero correlation**: Spearman r=0.010 (p=0.19), confirmed independent of normalization choice. Cause: WMT 2022 uk-en data is domain "other" (Telegram/personal messages, avg 12 words/sentence) vs. "news" for all other languages. BLEU/ChrF surface overlap with a single reference is a poor proxy for quality in conversational text with many valid paraphrases. This is a domain confound, not a language/script effect.
- **Added `domain` column** to `wmt_da.py`, `wmt_mqm.py`, and `run_correlation_analysis()` output so domain effects are visible in `correlations.csv`.
- **11/11 tests still pass** after correlation.py changes.

### Key finding: Ukrainian is a domain outlier
All other languages are evaluated on **news** text. Ukrainian WMT 2022 was collected from conversational sources (domain="other"). This means the alphabetic/Indo-European averages in Dr. Fulda's analysis are confounded by domain. Should be flagged in the paper — Ukrainian may need to be treated separately or excluded from family/script aggregations.

### SLURM status
- 11304485 (synthetic, m13l): RUNNING as of session end — will finish lo translation (~2h) then synthetic MQM scoring
- Next: once 11304485 completes, `sbatch scripts/slurm/run_comet.sh` (dw/matrix) for BERTScore/COMET/xCOMET

---

## 2026-04-06 (Session 2)

### What we did
- Fixed language family/script classification bugs: `he` moved to `afro_asiatic`, `ps` moved out of `afro_asiatic` (it's Indo-European/Iranian), `ha` moved out of `niger_congo` (it's Afro-Asiatic/Chadic), `lo` moved from `alphabetic` to `abugida` (Lao is Brahmic-derived like Khmer). Fixed in both `correlation.py` and `settings.toml`.
- Added `tier2_langs = ["sw", "ht", "lo"]` to settings.toml and updated pipeline.py to use it. Previously, Tier 2 FLORES/NLLB was running on all 6 low-resource langs (ha/km/ps unnecessarily, since they have DA data).
- Fixed SLURM scripts: `mamba activate` → `$SLURM_SUBMIT_DIR` + `.venv/bin/python` (old scripts failed with exit code 127)
- Fixed `_text_lists()` in pipeline.py: NaN values in text columns caused TypeError in sacrebleu
- Pre-downloaded NLLB-200 and Llama-3.1-8B weights on login node (network available)
- Submitted SLURM jobs: 11262595 (NLLB), 11262605 (synthetic, dependency)
- **Full login-node pipeline run succeeded**: 150,347 MQM segments (de/ru/zh) + 220,987 DA segments (cs/ha/km/ps/tr/uk) = 371,334 segments total

### First results (BLEU + ChrF, 9 languages)

Notable findings:
- **ChrF consistently outperforms BLEU** across all script types and families
- **Abugida scripts (km, lo)** show highest metric correlation (ChrF r=0.38) — counterintuitive relative to Dr. Fulda's hypothesis; may reflect km/lo having fewer segments (4.7k each) rather than script effects
- **Ukrainian (uk)** near-zero correlation for both metrics — suspicious; could be annotation noise, domain mismatch, or genuinely poor metric performance. Worth investigating separately.
- **Russian (ru)** BLEU correlation negative (-0.04) — unusually low compared to de (0.15) and zh (0.15)
- **Medium tier** outperforms high tier for both metrics — likely because high tier (de/ru/zh) spans multiple WMT years/domains with more variance

### SLURM job status
- 11262595 (run_nllb.sh): Pending/Running as of session end
- 11262605 (run_synthetic_mqm.sh): Pending, depends on 11262595

---

## 2026-04-06 (Session 1)

### What we did
- **Wired Dr. Fulda's analysis into pipeline.py**: Added calls to `summarize_by_script()`, `summarize_by_family()`, `plot_correlation_by_script_type()`, and `plot_correlation_by_language_family()` in `run_pipeline()`. Results saved to `script_type_summary.csv`, `family_summary.csv`, and the corresponding PNG plots.
- **Installed pytest** (was missing from venv): `uv pip install pytest`
- **Confirmed 11/11 unit tests pass** (`tests/test_converter.py`)
- **Smoke test (Tiers 1a + 1b)**: Confirmed working end-to-end on login node:
  - Tier 1a WMT MQM: 49,741 segments loaded for `de` ✓
  - Tier 1b WMT DA: 135,844 segments loaded for `cs` ✓
  - Tier 2 NLLB translation killed by login node CPU time limit (expected — belongs on SLURM)

### Key findings
- WMT DA has more languages than originally thought: `['bn', 'cs', 'de', 'et', 'fi', 'fr', 'gu', 'ha', 'hi', 'is', 'ja', 'kk', 'km', 'lt', 'lv', 'pl', 'ps', 'ru', 'ta', 'tr', 'uk', 'xh', 'zh', 'zu']` — future work could expand DA coverage further
- NLLB translation runs on CPU only on login node (no usable GPU); **must run `run_nllb.sh` via SLURM** for Tier 2 data generation

### Next steps
1. **Submit SLURM jobs** for Tier 2 data generation:
   - `sbatch scripts/slurm/run_nllb.sh` — translate FLORES+ for low-resource langs (sw, ht, lo)
   - `sbatch scripts/slurm/run_synthetic_mqm.sh` — LLM-annotate with GEMBA-MQM style prompting
2. **Run full pipeline** after caches are populated (BLEU + ChrF first, then add BERTScore/COMET on GPU)
3. **Update README** — reflect expanded language set (13 langs), script/family analysis dimensions, Dr. Fulda's research questions

---

## 2026-04-05

### What we did
- Full pivot from private LDS corpus to public data sources
- Rewrote data loaders: `wmt_mqm.py`, `wmt_da.py`, `flores.py`, `nllb.py`, `synthetic_mqm.py`
- Added `data/utils.py` (shared ISO→BCP47 map, JSONL cache helpers, WMT column renaming)
- Added `constants.py` with `AnnotationTier` class
- Rewrote `pipeline.py` for three-tier loading sequence
- Added language family and script type analysis to `correlation.py` and `plots.py` (Dr. Fulda's feedback)
- Expanded language set to 13 languages across 3 resource tiers
- Created SLURM scripts for GPU jobs

### Key discoveries
- `RicardoRei/wmt-mqm-human-evaluation` is sentence-level aggregated scores only (no spans/categories) — category analysis disabled until span data is available
- `facebook/flores` broken in modern datasets library — switched to `openlanguagedata/flores_plus`
- hr/ro not in WMT DA — replaced with cs/tr
- he-en not in MQM dataset — skipped gracefully
