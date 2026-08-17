#!/bin/bash
#SBATCH -J wickdet_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Load your cluster's conda/anaconda module before submitting this job." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wickdet-a100
bash ./run_smoke.sh --run-id smoke_${SLURM_JOB_ID} --resume
