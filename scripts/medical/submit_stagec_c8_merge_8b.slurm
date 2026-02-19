#!/bin/bash
#SBATCH --job-name=stagec-c8-8b
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm-stagec-c8-8b-%j.out
#SBATCH --error=slurm-stagec-c8-8b-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

export MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-8B}"
export C7_CKPT_ROOT="${C7_CKPT_ROOT:-${PROJECT_ROOT}/TruthRL/data/medical/stagec_c7_8b_full/sft_ckpt}"
export C8_ROOT="${C8_ROOT:-${PROJECT_ROOT}/TruthRL/data/medical/stagec_c8_8b_full}"
export USE_APPTAINER="${USE_APPTAINER:-1}"

bash scripts/medical/run_stagec_c8_merge_8b.sh
