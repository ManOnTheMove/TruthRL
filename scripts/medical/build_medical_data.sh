#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_SCRIPT_DIR="${REPO_ROOT}/data_utils/medical"

RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/data/medical/raw}"
PROCESSED_CSV_DIR="${PROCESSED_CSV_DIR:-${REPO_ROOT}/data/medical/processed_csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/medical/verl}"
REPORTS_DIR="${REPORTS_DIR:-${REPO_ROOT}/data/medical/reports}"
DISTILL_DIR="${DISTILL_DIR:-${REPO_ROOT}/data/medical/distill}"

SEED="${SEED:-2026}"
GRPO_EVAL_SPLIT="${GRPO_EVAL_SPLIT:-test}"
MEDMCQA_SPLITS="${MEDMCQA_SPLITS:-dev,test}"
DROP_MISSING_TARGET="${DROP_MISSING_TARGET:-true}"
REQUIRE_DISTILL="${REQUIRE_DISTILL:-true}"
PLACEHOLDER_MODE="${PLACEHOLDER_MODE:-simple}"
USE_APPTAINER="${USE_APPTAINER:-1}"

mkdir -p "${RAW_DATA_DIR}" "${PROCESSED_CSV_DIR}" "${OUTPUT_DIR}" "${REPORTS_DIR}"

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
    echo "Set APPTAINER_IMAGE to your truthrl.sif path, or USE_APPTAINER=0." >&2
    exit 1
  fi

  NODE_TMPDIR="${SLURM_TMPDIR:-/tmp}"
  if ! mkdir -p "${NODE_TMPDIR}" >/dev/null 2>&1; then
    NODE_TMPDIR="/tmp"
  fi
  CACHE_CANDIDATE="${APPTAINER_CACHEDIR:-${NODE_TMPDIR}/apptainer_cache_${USER}}"
  TMP_CANDIDATE="${APPTAINER_TMPDIR:-${NODE_TMPDIR}/apptainer_tmp_${USER}}"

  if ! mkdir -p "${CACHE_CANDIDATE}" >/dev/null 2>&1; then
    CACHE_CANDIDATE="/tmp/apptainer_cache_${USER}"
    mkdir -p "${CACHE_CANDIDATE}"
  fi
  if ! mkdir -p "${TMP_CANDIDATE}" >/dev/null 2>&1; then
    TMP_CANDIDATE="/tmp/apptainer_tmp_${USER}"
    mkdir -p "${TMP_CANDIDATE}"
  fi
  export APPTAINER_CACHEDIR="${CACHE_CANDIDATE}"
  export APPTAINER_TMPDIR="${TMP_CANDIDATE}"

  NV_FLAG=()
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG=(--nv)
  fi

  PY_CMD_PREFIX=(
    apptainer exec "${NV_FLAG[@]}" --cleanenv
    --env "PYTHONPATH=${REPO_ROOT}/training/verl"
    --env "RAW_DATA_DIR=${RAW_DATA_DIR}"
    --env "PROCESSED_CSV_DIR=${PROCESSED_CSV_DIR}"
    --env "OUTPUT_DIR=${OUTPUT_DIR}"
    --env "REPORTS_DIR=${REPORTS_DIR}"
    --env "DISTILL_DIR=${DISTILL_DIR}"
    --env "SEED=${SEED}"
    --env "GRPO_EVAL_SPLIT=${GRPO_EVAL_SPLIT}"
    --env "MEDMCQA_SPLITS=${MEDMCQA_SPLITS}"
    --env "DROP_MISSING_TARGET=${DROP_MISSING_TARGET}"
    --env "REQUIRE_DISTILL=${REQUIRE_DISTILL}"
    --env "PLACEHOLDER_MODE=${PLACEHOLDER_MODE}"
    "${IMAGE_PATH}"
    python3
  )
else
  PY_CMD_PREFIX=(python3)
fi

echo "[INFO] Stage B build settings:"
echo "[INFO] USE_APPTAINER=${USE_APPTAINER}"
echo "[INFO] RAW_DATA_DIR=${RAW_DATA_DIR}"
echo "[INFO] PROCESSED_CSV_DIR=${PROCESSED_CSV_DIR}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] REPORTS_DIR=${REPORTS_DIR}"
echo "[INFO] DISTILL_DIR=${DISTILL_DIR}"
echo "[INFO] REQUIRE_DISTILL=${REQUIRE_DISTILL}"

"${PY_CMD_PREFIX[@]}" "${PYTHON_SCRIPT_DIR}/build_medqa.py" \
  --raw_data_dir "${RAW_DATA_DIR}" \
  --processed_csv_dir "${PROCESSED_CSV_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --reports_dir "${REPORTS_DIR}" \
  --seed "${SEED}" \
  --grpo_eval_split "${GRPO_EVAL_SPLIT}"

if [[ "${DROP_MISSING_TARGET}" == "true" ]]; then
  DROP_FLAG=(--drop_missing_target)
else
  DROP_FLAG=(--no-drop_missing_target)
fi

"${PY_CMD_PREFIX[@]}" "${PYTHON_SCRIPT_DIR}/build_medmcqa.py" \
  --raw_data_dir "${RAW_DATA_DIR}" \
  --processed_csv_dir "${PROCESSED_CSV_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --reports_dir "${REPORTS_DIR}" \
  --splits "${MEDMCQA_SPLITS}" \
  "${DROP_FLAG[@]}"

if [[ "${REQUIRE_DISTILL}" == "true" ]]; then
  DISTILL_MODE_ARGS=(--require_distill)
else
  DISTILL_MODE_ARGS=(--no-require_distill --placeholder_mode "${PLACEHOLDER_MODE}")
fi

"${PY_CMD_PREFIX[@]}" "${PYTHON_SCRIPT_DIR}/build_distill_sft.py" \
  --sft_csv "${PROCESSED_CSV_DIR}/medqa_train_sft_half.csv" \
  --distill_dir "${DISTILL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --reports_dir "${REPORTS_DIR}" \
  --seed "${SEED}" \
  "${DISTILL_MODE_ARGS[@]}"

"${PY_CMD_PREFIX[@]}" "${PYTHON_SCRIPT_DIR}/validate_medical_data.py" validate \
  --rl-files \
    "${OUTPUT_DIR}/medqa_grpo_train.parquet" \
    "${OUTPUT_DIR}/medqa_grpo_test.parquet" \
    "${OUTPUT_DIR}/medmcqa_eval.parquet" \
  --sft-files \
    "${OUTPUT_DIR}/medqa_sft_train.parquet" \
    "${OUTPUT_DIR}/medqa_sft_val.parquet" \
  --report-json "${REPORTS_DIR}/stageB_data_report.json" \
  --report-md "${REPORTS_DIR}/stageB_data_report.md"

echo "[PASS] Stage B build pipeline completed."
echo "[PASS] Reports: ${REPORTS_DIR}/stageB_data_report.json, ${REPORTS_DIR}/stageB_data_report.md"

