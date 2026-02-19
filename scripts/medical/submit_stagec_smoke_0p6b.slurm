#!/bin/bash
#SBATCH --job-name=stagec-smoke-0p6b
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:n:1
#SBATCH --output=slurm-stagec-smoke-%j.out
#SBATCH --error=slurm-stagec-smoke-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

bash scripts/medical/run_stagec_pipeline_smoke_0p6b.sh
