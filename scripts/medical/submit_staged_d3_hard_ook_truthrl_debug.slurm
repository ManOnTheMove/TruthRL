#!/bin/bash
# Debug full-node smoke for D3 hard-OOK GRPO on Trillium.
# This validates the 4xH100 startup/training path before submitting production.
# Submit from trig-login01.
#SBATCH --job-name=stageD-d3-debug
#SBATCH --account=def-zshakeri
#SBATCH --partition=debug_full_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/erichyu/stageD_d3_hard_ook/debug_logs/%x-%j.out
#SBATCH --error=/scratch/erichyu/stageD_d3_hard_ook/debug_logs/%x-%j.err

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

mkdir -p /scratch/erichyu/stageD_d3_hard_ook/debug_logs
cd "${REPO_ROOT}"

echo "[INFO] D3 debug REPO_ROOT=${REPO_ROOT}"
echo "[INFO] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "[INFO] Running as $(id)"

export USE_APPTAINER=1
export NPROC_PER_NODE=4
export NNODES=1
export TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_train_with_ook_d26.parquet}"
export VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_test.parquet}"
export RUN_ROOT="${RUN_ROOT:-/scratch/erichyu/stageD_d3_hard_ook/debug_run}"
export APPTAINER_IMAGE="${APPTAINER_IMAGE:-${REPO_ROOT}/../envs/truthrl_apptainer/truthrl.sif}"
export PROJECT_NAME="${PROJECT_NAME:-stage-d-d3-debug}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-truthrl-hard-ook-d3-debug}"

# Keep the debug smoke short, but still execute the real Ray/FSDP/vLLM/training path.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
export ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

bash "${REPO_ROOT}/scripts/medical/run_staged_d3_hard_ook_truthrl.sh" "$@"
