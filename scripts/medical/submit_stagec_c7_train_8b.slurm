#!/bin/bash
#SBATCH --job-name=stagec-c7-8b
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --output=slurm-stagec-c7-8b-%j.out
#SBATCH --error=slurm-stagec-c7-8b-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

export MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-8B}"
export TRAIN_PARQUET="${TRAIN_PARQUET:-${PROJECT_ROOT}/TruthRL/data/medical/verl/medqa_sft_train.parquet}"
export VAL_PARQUET="${VAL_PARQUET:-${PROJECT_ROOT}/TruthRL/data/medical/verl/medqa_sft_val.parquet}"
export RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/TruthRL/data/medical/stagec_c7_8b_full}"
export USE_APPTAINER="${USE_APPTAINER:-1}"
export PROJECT_NAME="${PROJECT_NAME:-stagec-c7-full}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b}"
export REPORT_NAME="${REPORT_NAME:-stageC_c7_train_8b_summary.json}"

# Requested by user for this run: train 10 epochs on 8B with 4xH100.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export LR="${LR:-1e-5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

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
