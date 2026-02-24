#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_ID="${JOB_ID:-8946648}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

DATE_TAG="${DATE_TAG:-2026-02-21}"
OUT_JSON="${OUT_JSON:-/home/erichyu/projects/def-zshakeri/erichyu/embc/chat_with_llm/stage_reports/stage_c/stageC_full_retest_g17_${DATE_TAG}.json}"
OUT_MD="${OUT_MD:-/home/erichyu/projects/def-zshakeri/erichyu/embc/chat_with_llm/stage_reports/stage_c/stageC_full_retest_g17_${DATE_TAG}.md}"

SFT_VAL_PARQUET="${SFT_VAL_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_sft_val.parquet}"
GRPO_TEST_PARQUET="${GRPO_TEST_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_test.parquet}"
DISTILL_DIR="${DISTILL_DIR:-${REPO_ROOT}/data/medical/distill}"
BASE_MODEL="${BASE_MODEL:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-8B}"
FIX_MODEL="${FIX_MODEL:-${REPO_ROOT}/data/medical/stagec_c8_8b_full_fixalpha/merged_model}"
WINDOWS="${WINDOWS:-512,1024,1536,2048}"
TEMPLATE_WINDOW="${TEMPLATE_WINDOW:-1536}"
MAX_NEW_TOKENS_B="${MAX_NEW_TOKENS_B:-2048}"
TRUNC_THRESH="${TRUNC_THRESH:-0.5}"

APPTAINER_IMAGE="${APPTAINER_IMAGE:-/home/erichyu/projects/def-zshakeri/erichyu/envs/truthrl_apptainer/truthrl.sif}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "[ERROR] Missing Apptainer image: ${APPTAINER_IMAGE}"
  exit 1
fi

if [[ ! -f "${SFT_VAL_PARQUET}" ]]; then
  echo "[ERROR] Missing SFT val parquet: ${SFT_VAL_PARQUET}"
  exit 1
fi
if [[ ! -f "${GRPO_TEST_PARQUET}" ]]; then
  echo "[ERROR] Missing GRPO test parquet: ${GRPO_TEST_PARQUET}"
  exit 1
fi
if [[ ! -d "${DISTILL_DIR}" ]]; then
  echo "[ERROR] Missing distill dir: ${DISTILL_DIR}"
  exit 1
fi

if (( SAMPLE_SIZE < 50 )); then
  echo "[ERROR] SAMPLE_SIZE must be >= 50, got ${SAMPLE_SIZE}"
  exit 1
fi

echo "[INFO] Stage-C full retest on g17 job ${JOB_ID}"
echo "[INFO] sample_size=${SAMPLE_SIZE} windows=${WINDOWS}"
echo "[INFO] out_json=${OUT_JSON}"
echo "[INFO] out_md=${OUT_MD}"

srun --jobid "${JOB_ID}" --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc "
  set -euo pipefail
  module load apptainer >/dev/null 2>&1
  apptainer exec --nv --cleanenv --env PYTHONPATH='${REPO_ROOT}/training/verl' '${APPTAINER_IMAGE}' \
    python3 '${REPO_ROOT}/scripts/medical/stagec_full_retest.py' \
      --sft_val_parquet '${SFT_VAL_PARQUET}' \
      --grpo_test_parquet '${GRPO_TEST_PARQUET}' \
      --distill_dir '${DISTILL_DIR}' \
      --base_model_path '${BASE_MODEL}' \
      --fix_model_path '${FIX_MODEL}' \
      --sample_size '${SAMPLE_SIZE}' \
      --windows '${WINDOWS}' \
      --template_window '${TEMPLATE_WINDOW}' \
      --max_new_tokens_b '${MAX_NEW_TOKENS_B}' \
      --truncation_high_threshold '${TRUNC_THRESH}' \
      --gpu_memory_utilization '${GPU_MEM_UTIL}' \
      --max_model_len '${MAX_MODEL_LEN}' \
      --out_json '${OUT_JSON}' \
      --out_md '${OUT_MD}'
"

echo "[PASS] Stage-C full retest done."
