#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-0.6B}"
INPUT_SFT_CSV="${INPUT_SFT_CSV:-${REPO_ROOT}/data/medical/processed_csv/medqa_train_sft_half.csv}"
SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/data/medical/stagec_smoke_0p6b}"
DISTILL_DIR="${DISTILL_DIR:-${SMOKE_ROOT}/distill}"
VERL_DIR="${VERL_DIR:-${SMOKE_ROOT}/verl}"
REPORT_DIR="${REPORT_DIR:-${SMOKE_ROOT}/reports}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${SMOKE_ROOT}/sft_ckpt}"
MERGED_DIR="${MERGED_DIR:-${SMOKE_ROOT}/merged_model}"
SMOKE_CSV="${SMOKE_CSV:-${SMOKE_ROOT}/medqa_sft_smoke.csv}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-16}"
SERVER_PORT="${SERVER_PORT:-8010}"
USE_APPTAINER="${USE_APPTAINER:-1}"
FORCE_TARGET_BOXED="${FORCE_TARGET_BOXED:-1}"
SERVER_LOG="${SERVER_LOG:-${REPORT_DIR}/qwen0p6b_server.log}"
EXPECT_LORA_ALPHA="${EXPECT_LORA_ALPHA:-16}"

mkdir -p "${DISTILL_DIR}" "${VERL_DIR}" "${REPORT_DIR}" "${SFT_CKPT_DIR}" "${MERGED_DIR}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] MODEL_PATH invalid: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${INPUT_SFT_CSV}" ]]; then
  echo "[ERROR] Missing INPUT_SFT_CSV: ${INPUT_SFT_CSV}"
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
      module load apptainer
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
  if ! mkdir -p "${NODE_TMPDIR}" >/dev/null 2>&1; then
    NODE_TMPDIR="/tmp"
  fi
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

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[INFO] stopping local server pid=${SERVER_PID}"
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

stop_server_if_running() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[INFO] stopping local server before training pid=${SERVER_PID}"
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}

create_smoke_csv() {
  "${PY_CMD[@]}" - <<PY
import csv
from pathlib import Path

inp = Path(r"${INPUT_SFT_CSV}")
out = Path(r"${SMOKE_CSV}")
limit = int(${SAMPLE_LIMIT})

with inp.open("r", encoding="utf-8", newline="") as fin:
    reader = csv.DictReader(fin)
    rows = []
    for i, row in enumerate(reader):
        if i >= limit:
            break
        rows.append(row)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[INFO] wrote smoke csv: {out} rows={len(rows)}")
PY
}

start_server() {
  echo "[INFO] starting local 0.6B server on port ${SERVER_PORT}"
  "${PY_CMD[@]}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${SERVER_PORT}" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 4096 \
    --enforce-eager \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"

  for _ in $(seq 1 180); do
    if curl -s "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
      echo "[PASS] local server healthy"
      return 0
    fi
    sleep 2
  done

  echo "[ERROR] local server health check timeout"
  return 1
}

run_distill() {
  FORCE_FLAG=(--no-force_target_boxed)
  if [[ "${FORCE_TARGET_BOXED}" == "1" ]]; then
    FORCE_FLAG=(--force_target_boxed)
  fi

  "${PY_CMD[@]}" "${REPO_ROOT}/data_utils/medical/qwen_local_distill.py" \
    --sft_csv "${SMOKE_CSV}" \
    --save_dir "${DISTILL_DIR}" \
    --endpoint "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
    --model "${MODEL_PATH}" \
    --max_retries 3 \
    --timeout_s 120 \
    --max_tokens 768 \
    --report_json "${REPORT_DIR}/stageC_smoke_0p6b_report.json" \
    --failed_ids_path "${REPORT_DIR}/stageC_smoke_0p6b_failed_ids.txt" \
    --fail_on_error \
    "${FORCE_FLAG[@]}"
}

run_strict_rebuild() {
  "${PY_CMD[@]}" "${REPO_ROOT}/data_utils/medical/build_distill_sft.py" \
    --sft_csv "${SMOKE_CSV}" \
    --distill_dir "${DISTILL_DIR}" \
    --output_dir "${VERL_DIR}" \
    --reports_dir "${REPORT_DIR}" \
    --train_ratio 0.5 \
    --seed 2026 \
    --require_distill \
    --missing_report_path "${REPORT_DIR}/stageC_smoke_missing.txt"
}

run_validate() {
  "${PY_CMD[@]}" "${REPO_ROOT}/data_utils/medical/validate_medical_data.py" validate \
    --rl-files \
      "${REPO_ROOT}/data/medical/verl/medqa_grpo_train.parquet" \
      "${REPO_ROOT}/data/medical/verl/medqa_grpo_test.parquet" \
      "${REPO_ROOT}/data/medical/verl/medmcqa_eval.parquet" \
    --sft-files \
      "${VERL_DIR}/medqa_sft_train.parquet" \
      "${VERL_DIR}/medqa_sft_val.parquet" \
    --report-json "${REPORT_DIR}/stageC_smoke_data_report.json" \
    --report-md "${REPORT_DIR}/stageC_smoke_data_report.md"
}

run_sft_smoke() {
  "${PY_CMD[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${VERL_DIR}/medqa_sft_train.parquet" \
    data.val_files="${VERL_DIR}/medqa_sft_val.parquet" \
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
    trainer.project_name=stagec-smoke \
    trainer.experiment_name=qwen3-0p6b \
    trainer.total_epochs=1 \
    trainer.logger='["console"]' \
    trainer.save_freq=2 \
    trainer.test_freq=-1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable
}

run_merge_smoke() {
  local latest_ckpt
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
      --trust_remote_code
  else
    echo "[INFO] adapter_config.json not found. Fallback to verl.model_merger (fsdp backend)."
    "${PY_CMD[@]}" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${latest_ckpt}" \
      --target_dir "${MERGED_DIR}"

    local adapter_dir="${MERGED_DIR}/lora_adapter"
    local adapter_cfg="${adapter_dir}/adapter_config.json"
    if [[ ! -f "${adapter_cfg}" ]]; then
      echo "[ERROR] FSDP merge output missing adapter config: ${adapter_cfg}"
      exit 1
    fi
    "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/repair_lora_adapter_config.py" \
      --adapter-config "${adapter_cfg}" \
      --expected-lora-alpha "${EXPECT_LORA_ALPHA}"
    "${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/merge_lora.py" \
      --base_model_path "${MODEL_PATH}" \
      --lora_model_path "${adapter_dir}" \
      --merged_model_save_path "${MERGED_DIR}" \
      --trust_remote_code
  fi

  "${PY_CMD[@]}" - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir = r"${MERGED_DIR}"
_ = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
_ = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cpu")
print("[PASS] merged model load smoke ok")
PY
}

write_final_report() {
  "${PY_CMD[@]}" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

report = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_path": r"${MODEL_PATH}",
    "sample_limit": int(${SAMPLE_LIMIT}),
    "server_port": int(${SERVER_PORT}),
    "smoke_root": r"${SMOKE_ROOT}",
    "status": "pass",
    "outputs": {
        "distill_report": r"${REPORT_DIR}/stageC_smoke_0p6b_report.json",
        "data_report": r"${REPORT_DIR}/stageC_smoke_data_report.json",
        "sft_ckpt_dir": r"${SFT_CKPT_DIR}",
        "merged_dir": r"${MERGED_DIR}",
    },
}

out = Path(r"${REPORT_DIR}/stageC_smoke_pipeline_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[PASS] wrote final smoke report: {out}")
PY
}

echo "[INFO] Stage C smoke pipeline (0.6B) start"
create_smoke_csv
start_server
run_distill
stop_server_if_running
run_strict_rebuild
run_validate
run_sft_smoke
run_merge_smoke
write_final_report

echo "[PASS] Stage C 0.6B smoke pipeline completed"
