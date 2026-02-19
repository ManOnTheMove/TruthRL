#!/bin/bash
#SBATCH --job-name=stagec-c7c8-0p6b
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm-stagec-c7c8-0p6b-%j.out
#SBATCH --error=slurm-stagec-c7c8-0p6b-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

export MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-0.6B}"
export SAMPLE_TRAIN="${SAMPLE_TRAIN:-128}"
export SAMPLE_VAL="${SAMPLE_VAL:-32}"
export USE_APPTAINER="${USE_APPTAINER:-1}"

bash scripts/medical/run_stagec_c7c8_smoke_0p6b.sh
