#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_SCRIPT="${REPO_ROOT}/data_utils/medical/validate_medical_data.py"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/medical/verl}"
REPORTS_DIR="${REPORTS_DIR:-${REPO_ROOT}/data/medical/reports}"
USE_APPTAINER="${USE_APPTAINER:-1}"

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
      module load apptainer
    fi
  fi

  if ! command -v apptainer >/dev/null 2>&1; then
    echo "[ERROR] apptainer not found in PATH (set USE_APPTAINER=0 to run on host)." >&2
    exit 1
  fi

  if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "[ERROR] Apptainer image not found: ${IMAGE_PATH}" >&2
    exit 1
  fi

  NODE_TMPDIR="${SLURM_TMPDIR:-/tmp}"
  if ! mkdir -p "${NODE_TMPDIR}" >/dev/null 2>&1; then
    NODE_TMPDIR="/tmp"
  fi
  CACHE_CANDIDATE="${APPTAINER_CACHEDIR:-${NODE_TMPDIR}/apptainer_cache_${USER}}"
  TMP_CANDIDATE="${APPTAINER_TMPDIR:-${NODE_TMPDIR}/apptainer_tmp_${USER}}"
  mkdir -p "${CACHE_CANDIDATE}" "${TMP_CANDIDATE}"
  export APPTAINER_CACHEDIR="${CACHE_CANDIDATE}"
  export APPTAINER_TMPDIR="${TMP_CANDIDATE}"

  NV_FLAG=()
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG=(--nv)
  fi

  PY_CMD_PREFIX=(
    apptainer exec "${NV_FLAG[@]}" --cleanenv
    --env "PYTHONPATH=${REPO_ROOT}/training/verl"
    "${IMAGE_PATH}"
    python3
  )
else
  PY_CMD_PREFIX=(python3)
fi

echo "[INFO] Running Stage B medical data validation"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] REPORTS_DIR=${REPORTS_DIR}"

"${PY_CMD_PREFIX[@]}" "${PYTHON_SCRIPT}" validate \
  --rl-files \
    "${OUTPUT_DIR}/medqa_grpo_train.parquet" \
    "${OUTPUT_DIR}/medqa_grpo_test.parquet" \
    "${OUTPUT_DIR}/medmcqa_eval.parquet" \
  --sft-files \
    "${OUTPUT_DIR}/medqa_sft_train.parquet" \
    "${OUTPUT_DIR}/medqa_sft_val.parquet" \
  --report-json "${REPORTS_DIR}/stageB_data_report.json" \
  --report-md "${REPORTS_DIR}/stageB_data_report.md"

echo "[PASS] Validation succeeded."

