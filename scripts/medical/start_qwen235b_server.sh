#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-235B-A22B}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-8}"
QUANTIZATION="${QUANTIZATION:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
USE_APPTAINER="${USE_APPTAINER:-1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/medical}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen235b_server.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/qwen235b_server.pid}"
HEALTH_MAX_WAIT_S="${HEALTH_MAX_WAIT_S:-600}"
HEALTH_INTERVAL_S="${HEALTH_INTERVAL_S:-5}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
    echo "[ERROR] existing server process is running with pid=${old_pid}. Stop it first."
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found"
  exit 1
fi

GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
if [[ "${GPU_COUNT}" -lt "${TP_SIZE}" ]]; then
  echo "[ERROR] Not enough full GPUs: required TP_SIZE=${TP_SIZE}, found GPU_COUNT=${GPU_COUNT}"
  exit 2
fi

if nvidia-smi -L | grep -q "MIG"; then
  echo "[WARN] MIG devices detected. 235B deployment is expected on full GPUs."
fi

if [[ "${MODEL_PATH}" == /* ]]; then
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] MODEL_PATH looks local but config.json not found: ${MODEL_PATH}"
    exit 1
  fi
fi

if [[ "${USE_APPTAINER}" == "1" ]]; then
  if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    IMAGE_PATH="${APPTAINER_IMAGE}"
  elif [[ -f "${REPO_ROOT}/truthrl.sif" ]]; then
    IMAGE_PATH="${REPO_ROOT}/truthrl.sif"
  else
    IMAGE_PATH="$(cd "${REPO_ROOT}/../.." && pwd)/envs/truthrl_apptainer/truthrl.sif"
  fi

  if ! command -v apptainer >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
      module load apptainer >/dev/null 2>&1 || true
    fi
  fi

  if ! command -v apptainer >/dev/null 2>&1; then
    echo "[ERROR] apptainer not found"
    exit 1
  fi

  if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "[ERROR] Apptainer image not found: ${IMAGE_PATH}"
    exit 1
  fi

  NODE_TMPDIR="${SLURM_TMPDIR:-/tmp}"
  mkdir -p "${NODE_TMPDIR}" >/dev/null 2>&1 || true
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${NODE_TMPDIR}/apptainer_cache_${USER}}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${NODE_TMPDIR}/apptainer_tmp_${USER}}"
  mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

  NV_FLAG=()
  if nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG=(--nv)
  fi

  PY_CMD=(apptainer exec "${NV_FLAG[@]}" --cleanenv --env "PYTHONPATH=${REPO_ROOT}/training/verl" "${IMAGE_PATH}" python3)
else
  PY_CMD=(python3)
fi

EAGER_FLAG=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EAGER_FLAG=(--enforce-eager)
fi

QUANT_FLAG=()
if [[ -n "${QUANTIZATION}" ]]; then
  QUANT_FLAG=(--quantization "${QUANTIZATION}")
fi

echo "[INFO] Starting Qwen235B server"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] DTYPE=${DTYPE} TP_SIZE=${TP_SIZE} PORT=${SERVER_PORT}"
if [[ -n "${QUANTIZATION}" ]]; then
  echo "[INFO] QUANTIZATION=${QUANTIZATION}"
fi
echo "[INFO] LOG_FILE=${LOG_FILE}"

"${PY_CMD[@]}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  "${QUANT_FLAG[@]}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  "${EAGER_FLAG[@]}" \
  >"${LOG_FILE}" 2>&1 &

SERVER_PID="$!"
echo "${SERVER_PID}" > "${PID_FILE}"

echo "[INFO] waiting for health endpoint /v1/models"
max_tries=$((HEALTH_MAX_WAIT_S / HEALTH_INTERVAL_S))
if [[ "${max_tries}" -lt 1 ]]; then
  max_tries=1
fi

for _ in $(seq 1 "${max_tries}"); do
  if curl -s "http://${SERVER_HOST}:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[PASS] Qwen235B server healthy pid=${SERVER_PID}"
    exit 0
  fi

  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[ERROR] server process exited early. Tail logs:"
    tail -n 120 "${LOG_FILE}" || true
    rm -f "${PID_FILE}"

    if nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >/tmp/qwen235b_mem.$$ 2>/dev/null; then
      echo "[INFO] GPU process memory snapshot:"
      cat /tmp/qwen235b_mem.$$ || true
      rm -f /tmp/qwen235b_mem.$$
    fi
    exit 3
  fi

  sleep "${HEALTH_INTERVAL_S}"
done

echo "[ERROR] health check timeout after ${HEALTH_MAX_WAIT_S}s"
tail -n 120 "${LOG_FILE}" || true
if nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >/tmp/qwen235b_gpu.$$ 2>/dev/null; then
  echo "[INFO] GPU memory snapshot:"
  cat /tmp/qwen235b_gpu.$$ || true
  rm -f /tmp/qwen235b_gpu.$$
fi
exit 4
