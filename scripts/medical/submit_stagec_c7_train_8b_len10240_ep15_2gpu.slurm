#!/bin/bash
#SBATCH --job-name=stagec-c7-8b-len10240-ep15-2g
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output=slurm-stagec-c7-8b-len10240-ep15-2g-%j.out
#SBATCH --error=slurm-stagec-c7-8b-len10240-ep15-2g-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

export MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-8B}"
export TRAIN_PARQUET="${TRAIN_PARQUET:-${PROJECT_ROOT}/TruthRL/data/medical/verl/medqa_sft_train.parquet}"
export VAL_PARQUET="${VAL_PARQUET:-${PROJECT_ROOT}/TruthRL/data/medical/verl/medqa_sft_val.parquet}"
export RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/TruthRL/data/medical/stagec_c7_8b_len10240_ep15_2gpu}"
export USE_APPTAINER="${USE_APPTAINER:-1}"
export PROJECT_NAME="${PROJECT_NAME:-stagec-c7-retrain}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-len10240-ep15-2gpu}"
export REPORT_NAME="${REPORT_NAME:-stageC_c7_train_8b_len10240_ep15_2gpu_summary.json}"

# Keep the same training setup as 4-GPU run, except GPU count.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-10240}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export LR="${LR:-1e-5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NNODES="${NNODES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

# Resumable training defaults.
export RESUME_MODE="${RESUME_MODE:-auto}"
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-${LOG_DIR}/gpu_minute_avg_${SLURM_JOB_ID}.log}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-5}"
MONITOR_WINDOW_S="${MONITOR_WINDOW_S:-60}"

bash scripts/medical/monitor_gpu_minute_avg.sh "${GPU_MONITOR_LOG}" "${MONITOR_INTERVAL_S}" "${MONITOR_WINDOW_S}" &
MONITOR_PID="$!"

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
    kill "${MONITOR_PID}" || true
  fi
}
trap cleanup EXIT

bash scripts/medical/run_stagec_c7_train_0p6b.sh
