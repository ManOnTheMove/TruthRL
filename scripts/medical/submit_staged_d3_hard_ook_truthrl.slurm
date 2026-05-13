#!/bin/bash
# Submit from Trillium GPU login node (trig-login01).
# Runtime output should go to SCRATCH because HOME/PROJECT are read-only on compute nodes.
#SBATCH --job-name=stageD-d3-hard-ook
#SBATCH --account=def-zshakeri
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/erichyu/stageD_d3_hard_ook/logs/%x-%j.out
#SBATCH --error=/scratch/erichyu/stageD_d3_hard_ook/logs/%x-%j.err

set -euo pipefail

if [[ -n "${TRUTHRL_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${TRUTHRL_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] \
  && [[ -d "${SLURM_SUBMIT_DIR}/scripts/medical" ]] \
  && [[ -d "${SLURM_SUBMIT_DIR}/training/verl" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

mkdir -p /scratch/erichyu/stageD_d3_hard_ook/logs
cd "${REPO_ROOT}"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "[INFO] Running as $(id)"

export USE_APPTAINER=1
export NPROC_PER_NODE=4
export NNODES=1
export TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_train_with_ook_d26.parquet}"
export RUN_ROOT="${RUN_ROOT:-/scratch/erichyu/stageD_d3_hard_ook/run}"
export APPTAINER_IMAGE="${APPTAINER_IMAGE:-${REPO_ROOT}/../envs/truthrl_apptainer/truthrl.sif}"
export PROJECT_NAME="${PROJECT_NAME:-stage-d-d3}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-truthrl-hard-ook-d3}"

bash "${REPO_ROOT}/scripts/medical/run_staged_d3_hard_ook_truthrl.sh" "$@"
