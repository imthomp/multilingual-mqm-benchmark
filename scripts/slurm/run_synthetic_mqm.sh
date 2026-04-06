#!/bin/bash
#SBATCH --job-name=mqmbench_synth
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# BYU supercomputer: model weights must be pre-downloaded before this job runs.
# Uses the same model as metrics.gemba_model in settings.toml.
# On a login node: python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('<your-model>')"
# (also download the model weights similarly)
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

.venv/bin/python -c "
import logging
logging.basicConfig(level=logging.INFO)
from mqmbench.config import init_settings, settings
from mqmbench.data.flores import load_flores
from mqmbench.data.nllb import generate_hypotheses
from mqmbench.data.synthetic_mqm import generate_synthetic_mqm

init_settings('settings.toml')
tier2_langs = list(settings.data.tier2_langs)
flores_df = load_flores(lang_codes=tier2_langs)
flores_hyp_df = generate_hypotheses(
    flores_df,
    model_name=settings.data.nllb_model,
    cache_dir=settings.data.cache_dir,
)
generate_synthetic_mqm(
    flores_hyp_df,
    model_name=settings.metrics.gemba_model,
    cache_dir=settings.data.cache_dir,
)
print('Synthetic MQM caching complete.')
"
