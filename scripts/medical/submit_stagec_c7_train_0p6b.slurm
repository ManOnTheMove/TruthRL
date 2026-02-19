#!/bin/bash
#SBATCH --job-name=stagec-c7-0p6b
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm-stagec-c7-0p6b-%j.out
#SBATCH --error=slurm-stagec-c7-0p6b-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

export MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-0.6B}"
export USE_APPTAINER="${USE_APPTAINER:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export SAVE_FREQ="${SAVE_FREQ:-200}"
export LR="${LR:-1e-5}"

bash scripts/medical/run_stagec_c7_train_0p6b.sh
