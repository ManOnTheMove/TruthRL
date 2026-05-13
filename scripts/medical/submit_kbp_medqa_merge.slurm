#!/bin/bash
#SBATCH --job-name=stageD-d25-kbp-merge
#SBATCH --account=def-zshakeri
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --qos=normal
#SBATCH --output=/scratch/erichyu/kbp_test/%x-%j.out
#SBATCH --error=/scratch/erichyu/kbp_test/%x-%j.err

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/../models" ]] && [[ -d "${SLURM_SUBMIT_DIR}/scripts/medical" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${PROJECT_ROOT}/TruthRL"
mkdir -p /scratch/erichyu/kbp_test

bash scripts/medical/run_kbp_merge_and_postprocess.sh "$@"
