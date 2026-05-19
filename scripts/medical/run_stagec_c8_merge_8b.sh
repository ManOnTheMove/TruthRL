#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-8B}"
C7_CKPT_ROOT="${C7_CKPT_ROOT:-${REPO_ROOT}/data/medical/stagec_c7_8b_full/sft_ckpt}"
C8_ROOT="${C8_ROOT:-${REPO_ROOT}/data/medical/stagec_c8_8b_full}"
MERGED_DIR="${MERGED_DIR:-${C8_ROOT}/merged_model}"
REPORT_DIR="${REPORT_DIR:-${C8_ROOT}/reports}"
REPORT_JSON="${REPORT_JSON:-${REPORT_DIR}/stageC_c8_merge_8b_summary.json}"
USE_APPTAINER="${USE_APPTAINER:-1}"
EXPECT_LORA_ALPHA="${EXPECT_LORA_ALPHA:-16}"

mkdir -p "${MERGED_DIR}" "${REPORT_DIR}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] MODEL_PATH invalid: ${MODEL_PATH}"
  exit 1
fi

C7_CKPT="${C7_CKPT:-}"
if [[ -z "${C7_CKPT}" ]]; then
  C7_CKPT="$(ls -d "${C7_CKPT_ROOT}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
fi
if [[ -z "${C7_CKPT}" ]]; then
  echo "[ERROR] no C7 checkpoint found under ${C7_CKPT_ROOT}"
  exit 1
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
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG=(--nv)
  fi
  PY_CMD=(apptainer exec "${NV_FLAG[@]}" --cleanenv --env "PYTHONPATH=${REPO_ROOT}/training/verl" "${IMAGE_PATH}" python3)
else
  PY_CMD=(python3)
fi

echo "[INFO] Stage C8 merge (8B)"
echo "[INFO] C7_CKPT=${C7_CKPT}"
echo "[INFO] MERGED_DIR=${MERGED_DIR}"

MERGE_BACKEND="peft"
if [[ -f "${C7_CKPT}/adapter_config.json" ]]; then
  "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/merge_lora.py" \
    --base_model_path "${MODEL_PATH}" \
    --lora_model_path "${C7_CKPT}" \
    --merged_model_save_path "${MERGED_DIR}" \
    --trust_remote_code \
    --expected-lora-alpha "${EXPECT_LORA_ALPHA}"
else
  MERGE_BACKEND="fsdp"
  echo "[INFO] adapter_config.json not found, fallback to verl.model_merger (fsdp backend)."
  "${PY_CMD[@]}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${C7_CKPT}" \
    --target_dir "${MERGED_DIR}" \
    --lora-alpha "${EXPECT_LORA_ALPHA}"

  ADAPTER_DIR="${MERGED_DIR}/lora_adapter"
  ADAPTER_CFG="${ADAPTER_DIR}/adapter_config.json"
  if [[ ! -f "${ADAPTER_CFG}" ]]; then
    echo "[ERROR] FSDP merge output missing adapter config: ${ADAPTER_CFG}"
    exit 1
  fi

  echo "[INFO] Repairing adapter config alpha (expected=${EXPECT_LORA_ALPHA})"
  "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/repair_lora_adapter_config.py" \
    --adapter-config "${ADAPTER_CFG}" \
    --expected-lora-alpha "${EXPECT_LORA_ALPHA}"

  echo "[INFO] Re-merging base + repaired adapter to produce effective merged weights"
  "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/merge_lora.py" \
    --base_model_path "${MODEL_PATH}" \
    --lora_model_path "${ADAPTER_DIR}" \
    --merged_model_save_path "${MERGED_DIR}" \
    --trust_remote_code \
    --expected-lora-alpha "${EXPECT_LORA_ALPHA}"
  MERGE_BACKEND="fsdp+peft"
fi

echo "[INFO] Validating merged model with transformers"
"${PY_CMD[@]}" - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir = r"${MERGED_DIR}"
_ = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
_ = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cpu")
print("[PASS] transformers load check ok")
PY

echo "[INFO] Validating merged model with verl utils"
"${PY_CMD[@]}" - <<PY
from verl.utils.model import get_generation_config, get_huggingface_actor_config
model_dir = r"${MERGED_DIR}"
_ = get_huggingface_actor_config(model_dir, trust_remote_code=True)
_ = get_generation_config(model_dir, trust_remote_code=True)
print("[PASS] verl load check ok")
PY

"${PY_CMD[@]}" - <<PY
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

report = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "pass",
    "model_path": r"${MODEL_PATH}",
    "c7_ckpt": r"${C7_CKPT}",
    "merged_dir": r"${MERGED_DIR}",
    "merge_backend": r"${MERGE_BACKEND}",
    "expected_lora_alpha": float("${EXPECT_LORA_ALPHA}"),
    "transformers_load_check": "pass",
    "verl_load_check": "pass",
    "run_node": socket.gethostname(),
    "slurm_job_id": r"${SLURM_JOB_ID:-}",
}
out = Path(r"${REPORT_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[PASS] wrote report: {out}")
PY

echo "[PASS] Stage C8 merge completed."
