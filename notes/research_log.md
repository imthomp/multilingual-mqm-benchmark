# Research Log — Multilingual MQM Benchmark

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
