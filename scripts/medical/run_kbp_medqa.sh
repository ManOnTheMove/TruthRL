#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

is_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

print_command() {
  printf "[DRY-RUN] Command:"
  printf " %q" "$@"
  printf "\n"
}

INPUT_PARQUET="${INPUT_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_train.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/medical/kbp}"
RUN_ID="${RUN_ID:-kbp_20260301_stagec-c8-8b_medqa_grpo_train}"
SPLIT="${SPLIT:-medqa_grpo_train}"
QUESTION_LIMIT="${QUESTION_LIMIT:-0}"
QUESTION_RANK_START="${QUESTION_RANK_START:-}"
QUESTION_RANK_END="${QUESTION_RANK_END:-}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"

PROBES_PER_QUESTION="${PROBES_PER_QUESTION:-256}"
N_CHUNK="${N_CHUNK:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model}"
MODEL_ID="${MODEL_ID:-stagec-c8-8b-merged}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
STOP_STR="${STOP_STR:-</answer>}"
INCLUDE_STOP_STR_IN_OUTPUT="${INCLUDE_STOP_STR_IN_OUTPUT:-1}"
SEED_BASE="${SEED_BASE:-20260301}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
PART_ROWS="${PART_ROWS:-80000}"
RESUME="${RESUME:-0}"

USE_APPTAINER="${USE_APPTAINER:-1}"

if [[ ! -f "${INPUT_PARQUET}" ]]; then
  echo "[ERROR] Missing INPUT_PARQUET: ${INPUT_PARQUET}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Invalid MODEL_PATH: ${MODEL_PATH}" >&2
  exit 1
fi

if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
  mkdir -p "${OUTPUT_ROOT}"
fi

if [[ "${USE_APPTAINER}" == "1" ]]; then
  if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    IMAGE_PATH="${APPTAINER_IMAGE}"
  elif [[ -f "${REPO_ROOT}/truthrl.sif" ]]; then
    IMAGE_PATH="${REPO_ROOT}/truthrl.sif"
  else
    IMAGE_PATH="$(cd "${REPO_ROOT}/.." && pwd)/envs/truthrl_apptainer/truthrl.sif"
  fi

  if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
    if ! command -v apptainer >/dev/null 2>&1; then
      if command -v module >/dev/null 2>&1; then
        module load apptainer >/dev/null 2>&1 || true
      fi
    fi
    if ! command -v apptainer >/dev/null 2>&1; then
      echo "[ERROR] apptainer not found" >&2
      exit 1
    fi
    if [[ ! -f "${IMAGE_PATH}" ]]; then
      echo "[ERROR] Apptainer image not found: ${IMAGE_PATH}" >&2
      exit 1
    fi
  fi

  NODE_TMPDIR="${SLURM_TMPDIR:-/tmp}"
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${NODE_TMPDIR}/apptainer_cache_${USER}}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${NODE_TMPDIR}/apptainer_tmp_${USER}}"
  if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
    mkdir -p "${NODE_TMPDIR}" >/dev/null 2>&1 || true
    mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
  fi

  SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
  CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/.cache}"
  CONFIG_ROOT="${CONFIG_ROOT:-${SCRATCH_ROOT}/.config}"
  if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
    mkdir -p "${CACHE_ROOT}/huggingface/modules" "${CACHE_ROOT}/flashinfer" "${CONFIG_ROOT}/vllm"
  fi

  export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CONFIG_ROOT}}"
  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"

  if [[ -z "${APPTAINER_BINDPATH:-}" ]]; then
    export APPTAINER_BINDPATH="/project:/project,/home/${USER}:/home/${USER},/scratch:/scratch,${CACHE_ROOT}:/home/${USER}/.cache,${CONFIG_ROOT}:/home/${USER}/.config"
  fi

  NV_FLAG=()
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG=(--nv)
  fi

  BIND_FLAG=()
  if [[ -n "${APPTAINER_BINDPATH:-}" ]]; then
    BIND_FLAG=(--bind "${APPTAINER_BINDPATH}")
  fi

  CONTAINER_ENVS="PYTHONPATH=${REPO_ROOT}/training/verl,XDG_CONFIG_HOME=/home/${USER}/.config,HF_HOME=/home/${USER}/.cache/huggingface,HF_MODULES_CACHE=/home/${USER}/.cache/huggingface/modules"
  PY_CMD=(apptainer exec "${NV_FLAG[@]}" "${BIND_FLAG[@]}" --cleanenv --env "${CONTAINER_ENVS}" "${IMAGE_PATH}" python3)
else
  PY_CMD=(python3)
fi

RESUME_FLAG=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

QUESTION_RANGE_FLAGS=()
if [[ -n "${QUESTION_RANK_START}" ]]; then
  QUESTION_RANGE_FLAGS+=(--question_rank_start "${QUESTION_RANK_START}")
fi
if [[ -n "${QUESTION_RANK_END}" ]]; then
  QUESTION_RANGE_FLAGS+=(--question_rank_end "${QUESTION_RANK_END}")
fi

POSTPROCESS_FLAG=()
if [[ "${POSTPROCESS_ONLY}" == "1" ]]; then
  POSTPROCESS_FLAG=(--postprocess_only)
fi

INCLUDE_STOP_FLAG=()
if [[ "${INCLUDE_STOP_STR_IN_OUTPUT}" == "1" ]]; then
  INCLUDE_STOP_FLAG=(--include_stop_str_in_output)
fi

echo "[INFO] Stage D2.5 KBP run"
echo "[INFO] RUN_ID=${RUN_ID} SPLIT=${SPLIT} QUESTION_LIMIT=${QUESTION_LIMIT} QUESTION_RANK_START=${QUESTION_RANK_START:-NA} QUESTION_RANK_END=${QUESTION_RANK_END:-NA} POSTPROCESS_ONLY=${POSTPROCESS_ONLY}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SAMPLING: probes=${PROBES_PER_QUESTION} n_chunk=${N_CHUNK} temp=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} min_p=${MIN_P} max_new_tokens=${MAX_NEW_TOKENS}"
echo "[INFO] RUNTIME: workers=${NUM_WORKERS} gpu_ids=${GPU_IDS} gpu_mem_util=${GPU_MEMORY_UTILIZATION} max_num_seqs=${MAX_NUM_SEQS} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} max_model_len=${MAX_MODEL_LEN}"

KBP_CMD=("${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/kbp_medqa_collect.py" \
  --input_parquet "${INPUT_PARQUET}" \
  --output_root "${OUTPUT_ROOT}" \
  --run_id "${RUN_ID}" \
  --split "${SPLIT}" \
  --question_limit "${QUESTION_LIMIT}" \
  "${QUESTION_RANGE_FLAGS[@]}" \
  --probes_per_question "${PROBES_PER_QUESTION}" \
  --n_chunk "${N_CHUNK}" \
  --num_workers "${NUM_WORKERS}" \
  --gpu_ids "${GPU_IDS}" \
  --model_path "${MODEL_PATH}" \
  --model_id "${MODEL_ID}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --min_p "${MIN_P}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --stop_str "${STOP_STR}" \
  "${INCLUDE_STOP_FLAG[@]}" \
  --seed_base "${SEED_BASE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --max_num_seqs "${MAX_NUM_SEQS}" \
  --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --part_rows "${PART_ROWS}" \
  "${RESUME_FLAG[@]}" \
  "${POSTPROCESS_FLAG[@]}" \
  "$@")

if is_truthy "${DRY_RUN}" || is_truthy "${PREFLIGHT_ONLY}"; then
  print_command "${KBP_CMD[@]}"
  echo "[DRY-RUN] Stage D2.5 KBP command was not executed."
  exit 0
fi

"${KBP_CMD[@]}"

echo "[PASS] Stage D2.5 KBP run completed."
