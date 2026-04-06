#!/bin/bash
#SBATCH --job-name=mqmbench_nllb
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# BYU supercomputer: model weights must be pre-downloaded before this job runs.
# On a login node: python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

.venv/bin/python -c "
import logging
logging.basicConfig(level=logging.INFO)
from mqmbench.config import init_settings, settings
from mqmbench.data.flores import load_flores
from mqmbench.data.nllb import generate_hypotheses

init_settings('settings.toml')
tier2_langs = list(settings.data.tier2_langs)
flores_df = load_flores(lang_codes=tier2_langs)
generate_hypotheses(
    flores_df,
    model_name=settings.data.nllb_model,
    cache_dir=settings.data.cache_dir,
)
print('NLLB hypothesis caching complete.')
"
