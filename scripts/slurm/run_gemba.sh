#!/bin/bash --login

#SBATCH --job-name=mqmbench_gemba
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=2
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90
#SBATCH --output=mqmbench_gemba_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export HF_HUB_OFFLINE=1

echo "Running GEMBA-MQM evaluation..."

mamba activate mqmbench

nvidia-smi || echo "nvidia-smi not available"

srun python3 scripts/run_pipeline.py --settings settings.toml
