#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-0.6B}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_sft_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_sft_val.parquet}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/data/medical/stagec_c7_0p6b_full}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${RUN_ROOT}/sft_ckpt}"
REPORT_DIR="${REPORT_DIR:-${RUN_ROOT}/reports}"
REPORT_NAME="${REPORT_NAME:-stageC_c7_train_0p6b_summary.json}"
REPORT_JSON="${REPORT_JSON:-${REPORT_DIR}/${REPORT_NAME}}"

USE_APPTAINER="${USE_APPTAINER:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_FREQ="${SAVE_FREQ:-200}"
LR="${LR:-1e-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
PROJECT_NAME="${PROJECT_NAME:-stagec-c7-full}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-0p6b}"

mkdir -p "${SFT_CKPT_DIR}" "${REPORT_DIR}"

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
  PY_ENV="PYTHONPATH=${REPO_ROOT}/training/verl,NPROC_PER_NODE=${NPROC_PER_NODE},NNODES=${NNODES},OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  PY_CMD=(apptainer exec "${NV_FLAG[@]}" --cleanenv --env "${PY_ENV}" "${IMAGE_PATH}" python3)
else
  PY_CMD=(python3)
fi

echo "[INFO] Stage C7 full SFT training"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TRAIN_PARQUET=${TRAIN_PARQUET}"
echo "[INFO] VAL_PARQUET=${VAL_PARQUET}"
echo "[INFO] SFT_CKPT_DIR=${SFT_CKPT_DIR}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE} NNODES=${NNODES}"

"${PY_CMD[@]}" - <<PY
import os
import torch

required = int(os.environ.get("NPROC_PER_NODE", "1"))
count = torch.cuda.device_count()
print(f"[INFO] torch.cuda.device_count={count} required={required}")
if count < required:
    raise SystemExit(f"Not enough visible GPUs: required={required}, found={count}")
PY

if (( NPROC_PER_NODE > 1 )); then
  ddp_check_script="$(mktemp "${TMPDIR:-/tmp}/stagec_ddp_check_XXXX.py")"
  cat > "${ddp_check_script}" <<'PY'
import os
import socket
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world = dist.get_world_size()
t = torch.tensor([1.0], device=f"cuda:{local_rank}")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
print(
    f"[DDP-CHECK] host={socket.gethostname()} rank={rank}/{world} "
    f"local_rank={local_rank} dev={torch.cuda.get_device_name(local_rank)} "
    f"all_reduce_sum={t.item():.1f}",
    flush=True,
)
dist.barrier()
dist.destroy_process_group()
PY
  echo "[INFO] Running DDP/FSDP preflight check across ${NPROC_PER_NODE} GPUs..."
  "${PY_CMD[@]}" -m torch.distributed.run --standalone --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" "${ddp_check_script}"
  rm -f "${ddp_check_script}"
fi

"${PY_CMD[@]}" -m torch.distributed.run --standalone --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=right \
  model.partial_pretrain="${MODEL_PATH}" \
  model.trust_remote_code=True \
  model.enable_gradient_checkpointing=True \
  model.fsdp_config.model_dtype=bf16 \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  model.strategy=fsdp \
  optim.lr="${LR}" \
  trainer.default_local_dir="${SFT_CKPT_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.logger='["console"]' \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.resume_mode=disable

latest_ckpt="$(ls -d "${SFT_CKPT_DIR}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -z "${latest_ckpt}" ]]; then
  echo "[ERROR] no checkpoint found under ${SFT_CKPT_DIR}"
  exit 1
fi

"${PY_CMD[@]}" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

report = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "pass",
    "model_path": r"${MODEL_PATH}",
    "train_parquet": r"${TRAIN_PARQUET}",
    "val_parquet": r"${VAL_PARQUET}",
    "sft_ckpt_dir": r"${SFT_CKPT_DIR}",
    "latest_ckpt": r"${latest_ckpt}",
    "total_epochs": int("${TOTAL_EPOCHS}"),
    "train_batch_size": int("${TRAIN_BATCH_SIZE}"),
    "micro_batch_size": int("${MICRO_BATCH_SIZE}"),
    "max_length": int("${MAX_LENGTH}"),
    "learning_rate": float("${LR}"),
    "nproc_per_node": int("${NPROC_PER_NODE}"),
    "nnodes": int("${NNODES}"),
}
out = Path(r"${REPORT_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[PASS] wrote report: {out}")
PY

echo "[PASS] Stage C7 full training completed."
