#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-0.6B}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_sft_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_sft_val.parquet}"
SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/data/medical/stagec_c7c8_0p6b_smoke}"
SMOKE_TRAIN_PARQUET="${SMOKE_TRAIN_PARQUET:-${SMOKE_ROOT}/medqa_sft_train_smoke.parquet}"
SMOKE_VAL_PARQUET="${SMOKE_VAL_PARQUET:-${SMOKE_ROOT}/medqa_sft_val_smoke.parquet}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${SMOKE_ROOT}/sft_ckpt}"
MERGED_DIR="${MERGED_DIR:-${SMOKE_ROOT}/merged_model}"
REPORT_DIR="${REPORT_DIR:-${SMOKE_ROOT}/reports}"
REPORT_JSON="${REPORT_JSON:-${REPORT_DIR}/stageC_c7c8_smoke_0p6b_summary.json}"
SAMPLE_TRAIN="${SAMPLE_TRAIN:-128}"
SAMPLE_VAL="${SAMPLE_VAL:-32}"
USE_APPTAINER="${USE_APPTAINER:-1}"
EXPECT_LORA_ALPHA="${EXPECT_LORA_ALPHA:-16}"

mkdir -p "${SMOKE_ROOT}" "${SFT_CKPT_DIR}" "${MERGED_DIR}" "${REPORT_DIR}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] MODEL_PATH invalid: ${MODEL_PATH}"
  exit 1
fi
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "[ERROR] Missing TRAIN_PARQUET: ${TRAIN_PARQUET}"
  exit 1
fi
if [[ ! -f "${VAL_PARQUET}" ]]; then
  echo "[ERROR] Missing VAL_PARQUET: ${VAL_PARQUET}"
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

echo "[INFO] Building C7/C8 smoke parquet subsets"
"${PY_CMD[@]}" - <<PY
from pathlib import Path
import pyarrow.parquet as pq

train_in = Path(r"${TRAIN_PARQUET}")
val_in = Path(r"${VAL_PARQUET}")
train_out = Path(r"${SMOKE_TRAIN_PARQUET}")
val_out = Path(r"${SMOKE_VAL_PARQUET}")
sample_train = int("${SAMPLE_TRAIN}")
sample_val = int("${SAMPLE_VAL}")

train_tbl = pq.read_table(train_in).slice(0, sample_train)
val_tbl = pq.read_table(val_in).slice(0, sample_val)

train_out.parent.mkdir(parents=True, exist_ok=True)
val_out.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(train_tbl, train_out, compression="snappy")
pq.write_table(val_tbl, val_out, compression="snappy")

print(f"[INFO] smoke train rows={train_tbl.num_rows} -> {train_out}")
print(f"[INFO] smoke val rows={val_tbl.num_rows} -> {val_out}")
PY

echo "[INFO] C7 smoke SFT training (Qwen3-0.6B, 1xH100)"
"${PY_CMD[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${SMOKE_TRAIN_PARQUET}" \
  data.val_files="${SMOKE_VAL_PARQUET}" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=256 \
  data.truncation=right \
  model.partial_pretrain="${MODEL_PATH}" \
  model.trust_remote_code=True \
  model.enable_gradient_checkpointing=True \
  model.fsdp_config.model_dtype=bf16 \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  model.strategy=fsdp \
  optim.lr=1e-5 \
  trainer.default_local_dir="${SFT_CKPT_DIR}" \
  trainer.project_name=stagec-c7c8-smoke \
  trainer.experiment_name=qwen3-0p6b \
  trainer.total_epochs=1 \
  trainer.logger='["console"]' \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.resume_mode=disable

echo "[INFO] C8 smoke merge"
latest_ckpt="$(ls -d "${SFT_CKPT_DIR}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -z "${latest_ckpt}" ]]; then
  echo "[ERROR] no LoRA checkpoint found under ${SFT_CKPT_DIR}"
  exit 1
fi

if [[ -f "${latest_ckpt}/adapter_config.json" ]]; then
  "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/merge_lora.py" \
    --base_model_path "${MODEL_PATH}" \
    --lora_model_path "${latest_ckpt}" \
    --merged_model_save_path "${MERGED_DIR}" \
    --trust_remote_code \
    --expected-lora-alpha "${EXPECT_LORA_ALPHA}"
else
  echo "[INFO] adapter_config.json not found, fallback to verl.model_merger (fsdp backend)."
  "${PY_CMD[@]}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${latest_ckpt}" \
    --target_dir "${MERGED_DIR}" \
    --lora-alpha "${EXPECT_LORA_ALPHA}"

  ADAPTER_DIR="${MERGED_DIR}/lora_adapter"
  ADAPTER_CFG="${ADAPTER_DIR}/adapter_config.json"
  if [[ ! -f "${ADAPTER_CFG}" ]]; then
    echo "[ERROR] FSDP merge output missing adapter config: ${ADAPTER_CFG}"
    exit 1
  fi
  "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/repair_lora_adapter_config.py" \
    --adapter-config "${ADAPTER_CFG}" \
    --expected-lora-alpha "${EXPECT_LORA_ALPHA}"
  "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/merge_lora.py" \
    --base_model_path "${MODEL_PATH}" \
    --lora_model_path "${ADAPTER_DIR}" \
    --merged_model_save_path "${MERGED_DIR}" \
    --trust_remote_code \
    --expected-lora-alpha "${EXPECT_LORA_ALPHA}"
fi

echo "[INFO] Validating merged model loadability"
"${PY_CMD[@]}" - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir = r"${MERGED_DIR}"
_ = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
_ = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cpu")
print("[PASS] merged model load smoke ok")
PY

"${PY_CMD[@]}" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

report = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "pass",
    "model_path": r"${MODEL_PATH}",
    "smoke_train_parquet": r"${SMOKE_TRAIN_PARQUET}",
    "smoke_val_parquet": r"${SMOKE_VAL_PARQUET}",
    "sft_ckpt_dir": r"${SFT_CKPT_DIR}",
    "latest_ckpt": r"${latest_ckpt}",
    "merged_dir": r"${MERGED_DIR}",
    "sample_train": int("${SAMPLE_TRAIN}"),
    "sample_val": int("${SAMPLE_VAL}"),
}
out = Path(r"${REPORT_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[PASS] wrote report: {out}")
PY

echo "[PASS] Stage C C7/C8 smoke pipeline completed."
