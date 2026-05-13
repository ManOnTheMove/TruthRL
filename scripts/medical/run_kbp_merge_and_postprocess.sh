#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/medical/kbp}"
FINAL_RUN_ID="${FINAL_RUN_ID:?FINAL_RUN_ID is required}"
BASE_RUN_ID="${BASE_RUN_ID:-}"
SHARD_RUN_IDS="${SHARD_RUN_IDS:?SHARD_RUN_IDS is required}"
SPLIT="${SPLIT:-medqa_grpo_train}"
QUESTION_LIMIT="${QUESTION_LIMIT:-0}"
OVERWRITE_FINAL="${OVERWRITE_FINAL:-1}"

USE_APPTAINER="${USE_APPTAINER:-1}"

if [[ "${USE_APPTAINER}" == "1" ]]; then
  if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    IMAGE_PATH="${APPTAINER_IMAGE}"
  elif [[ -f "${REPO_ROOT}/truthrl.sif" ]]; then
    IMAGE_PATH="${REPO_ROOT}/truthrl.sif"
  else
    IMAGE_PATH="$(cd "${REPO_ROOT}/.." && pwd)/envs/truthrl_apptainer/truthrl.sif"
  fi

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

  SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
  CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/.cache}"
  CONFIG_ROOT="${CONFIG_ROOT:-${SCRATCH_ROOT}/.config}"
  mkdir -p "${CACHE_ROOT}/huggingface/modules" "${CACHE_ROOT}/flashinfer" "${CONFIG_ROOT}/vllm"

  export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CONFIG_ROOT}}"
  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"

  if [[ -z "${APPTAINER_BINDPATH:-}" ]]; then
    export APPTAINER_BINDPATH="/project:/project,/home/${USER}:/home/${USER},/scratch:/scratch,${CACHE_ROOT}:/home/${USER}/.cache,${CONFIG_ROOT}:/home/${USER}/.config"
  fi

  PY_CMD=(apptainer exec --bind "${APPTAINER_BINDPATH}" --cleanenv --env "PYTHONPATH=${REPO_ROOT}/training/verl,XDG_CONFIG_HOME=/home/${USER}/.config,HF_HOME=/home/${USER}/.cache/huggingface,HF_MODULES_CACHE=/home/${USER}/.cache/huggingface/modules" "${IMAGE_PATH}" python3)
else
  PY_CMD=(python3)
fi

MERGE_ARGS=(
  --output_root "${OUTPUT_ROOT}"
  --split "${SPLIT}"
  --final_run_id "${FINAL_RUN_ID}"
  --shard_run_ids "${SHARD_RUN_IDS}"
)
if [[ -n "${BASE_RUN_ID}" ]]; then
  MERGE_ARGS+=(--base_run_id "${BASE_RUN_ID}")
fi
if [[ "${OVERWRITE_FINAL}" == "1" ]]; then
  MERGE_ARGS+=(--overwrite_final)
fi

echo "[INFO] merge shards: FINAL_RUN_ID=${FINAL_RUN_ID} BASE_RUN_ID=${BASE_RUN_ID:-none} SHARD_RUN_IDS=${SHARD_RUN_IDS}"
"${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/kbp_merge_shards.py" "${MERGE_ARGS[@]}"

echo "[INFO] run postprocess-only"
export RUN_ID="${FINAL_RUN_ID}"
export RESUME=1
export POSTPROCESS_ONLY=1
export OUTPUT_ROOT
export SPLIT
export QUESTION_LIMIT

bash "${REPO_ROOT}/scripts/medical/run_kbp_medqa.sh"
