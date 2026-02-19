#!/bin/bash
#SBATCH --job-name=qwen235b-c4c5
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --output=slurm-qwen235b-server-%j.out
#SBATCH --error=slurm-qwen235b-server-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

# Default to locally downloaded 235B model; can still override at submission time.
export MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-235B-A22B}"
export TP_SIZE="${TP_SIZE:-8}"
export DTYPE="${DTYPE:-bfloat16}"
export SERVER_PORT="${SERVER_PORT:-8000}"
export USE_APPTAINER="${USE_APPTAINER:-1}"
export HEALTH_MAX_WAIT_S="${HEALTH_MAX_WAIT_S:-5400}"
export HEALTH_INTERVAL_S="${HEALTH_INTERVAL_S:-10}"

# C5 auto-run controls.
export AUTO_RUN_C5="${AUTO_RUN_C5:-1}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
export MAX_PASSES="${MAX_PASSES:-4}"
export MAX_TOKENS="${MAX_TOKENS:-1024}"

cleanup() {
  bash scripts/medical/stop_qwen235b_server.sh || true
}
trap cleanup EXIT

echo "[INFO] C3 precheck: full GPU environment"
bash scripts/medical/check_full_gpu_env.sh

echo "[INFO] C4: start local Qwen3-235B-A22B server"
bash scripts/medical/start_qwen235b_server.sh

if [[ "${AUTO_RUN_C5}" == "1" ]]; then
  echo "[INFO] C5: run distillation automatically after C4 health check"
  bash scripts/medical/run_qwen235b_distill.sh
else
  echo "[INFO] AUTO_RUN_C5=0, keep server alive only"
  PID_FILE="${PROJECT_ROOT}/TruthRL/logs/medical/qwen235b_server.pid"
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]]; then
      wait "${pid}" || true
    fi
  fi
fi
