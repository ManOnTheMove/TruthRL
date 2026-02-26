#!/bin/bash
#SBATCH --job-name=stageD-d1-grpo
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

export USE_APPTAINER=1
export NPROC_PER_NODE=1
export NNODES=1

bash scripts/medical/run_staged_d1_grpo_truthrl.sh "$@"
