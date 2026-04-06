#!/bin/bash --login

#SBATCH --job-name=mqmbench_comet
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90
#SBATCH --output=mqmbench_comet_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

nvidia-smi || echo "nvidia-smi not available"

srun .venv/bin/python scripts/run_pipeline.py --settings settings.toml
