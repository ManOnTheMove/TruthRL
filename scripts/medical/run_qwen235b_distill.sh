#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SFT_CSV="${SFT_CSV:-${REPO_ROOT}/data/medical/processed_csv/medqa_train_sft_half.csv}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/data/medical/distill}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/data/medical/reports}"
PASS_REPORT_DIR="${PASS_REPORT_DIR:-${REPORT_DIR}/stageC_distill_passes_${SLURM_JOB_ID:-manual}}"
FINAL_REPORT_JSON="${FINAL_REPORT_JSON:-${REPORT_DIR}/stageC_distill_stats.json}"
FINAL_FAILED_IDS_PATH="${FINAL_FAILED_IDS_PATH:-${REPORT_DIR}/stageC_distill_failed_ids.txt}"

SERVER_PORT="${SERVER_PORT:-8000}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${SERVER_PORT}/v1/chat/completions}"
DISTILL_MODEL="${DISTILL_MODEL:-${MODEL_PATH:-/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-235B-A22B}}"
API_KEY="${API_KEY:-token-abc123}"

NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_PASSES="${MAX_PASSES:-4}"
PASS_SLEEP_S="${PASS_SLEEP_S:-20}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TIMEOUT_S="${TIMEOUT_S:-300}"
MAX_RETRIES="${MAX_RETRIES:-4}"
BACKOFF_S="${BACKOFF_S:-3}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
FORCE_TARGET_BOXED="${FORCE_TARGET_BOXED:-1}"
USE_APPTAINER="${USE_APPTAINER:-1}"

mkdir -p "${SAVE_DIR}" "${REPORT_DIR}" "${PASS_REPORT_DIR}"

if [[ ! -f "${SFT_CSV}" ]]; then
  echo "[ERROR] Missing SFT_CSV: ${SFT_CSV}"
  exit 1
fi

if ! [[ "${NUM_SHARDS}" =~ ^[0-9]+$ ]] || [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "[ERROR] NUM_SHARDS must be a positive integer"
  exit 1
fi

if ! [[ "${MAX_PASSES}" =~ ^[0-9]+$ ]] || [[ "${MAX_PASSES}" -lt 1 ]]; then
  echo "[ERROR] MAX_PASSES must be a positive integer"
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

if ! curl -s "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
  echo "[ERROR] Local teacher endpoint not healthy: http://127.0.0.1:${SERVER_PORT}/v1/models"
  exit 1
fi

FORCE_FLAG=(--no-force_target_boxed)
if [[ "${FORCE_TARGET_BOXED}" == "1" ]]; then
  FORCE_FLAG=(--force_target_boxed)
fi

run_one_pass() {
  local pass_id="$1"
  local rc=0
  local shard_pid
  local pids=()

  echo "[INFO] ===== Distill pass ${pass_id}/${MAX_PASSES} ====="
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_report="${PASS_REPORT_DIR}/pass_${pass_id}_shard_${shard}.json"
    local shard_failed="${PASS_REPORT_DIR}/pass_${pass_id}_shard_${shard}_failed.txt"

    "${PY_CMD[@]}" "${REPO_ROOT}/data_utils/medical/qwen_local_distill.py" \
      --sft_csv "${SFT_CSV}" \
      --save_dir "${SAVE_DIR}" \
      --endpoint "${ENDPOINT}" \
      --model "${DISTILL_MODEL}" \
      --api_key "${API_KEY}" \
      --timeout_s "${TIMEOUT_S}" \
      --max_retries "${MAX_RETRIES}" \
      --backoff_s "${BACKOFF_S}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --max_tokens "${MAX_TOKENS}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${shard}" \
      --report_json "${shard_report}" \
      --failed_ids_path "${shard_failed}" \
      --no-overwrite \
      --no-fail_on_error \
      "${FORCE_FLAG[@]}" &
    shard_pid="$!"
    pids+=("${shard_pid}")
  done

  for shard_pid in "${pids[@]}"; do
    wait "${shard_pid}" || rc=1
  done

  if [[ "${rc}" -ne 0 ]]; then
    echo "[WARN] Some shard workers exited non-zero in pass ${pass_id}"
  fi
}

summarize_missing() {
  local pass_id="$1"
  local pass_missing_file="${PASS_REPORT_DIR}/pass_${pass_id}_missing_ids.txt"
  local pass_summary_json="${PASS_REPORT_DIR}/pass_${pass_id}_summary.json"

  "${PY_CMD[@]}" - <<PY
import csv
import glob
import json
from pathlib import Path

sft_csv = Path(r"${SFT_CSV}")
save_dir = Path(r"${SAVE_DIR}")
missing_file = Path(r"${pass_missing_file}")
summary_file = Path(r"${pass_summary_json}")
pass_id = int("${pass_id}")

stat_keys = [
    "total_rows",
    "selected_rows",
    "attempted_rows",
    "success_rows",
    "skipped_existing_rows",
    "failed_rows",
    "api_fail_rows",
    "contract_fail_rows",
]
aggregate = {k: 0 for k in stat_keys}
reports = sorted(glob.glob(r"${PASS_REPORT_DIR}/pass_${pass_id}_shard_*.json"))
for rp in reports:
    with open(rp, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats = data.get("stats", {})
    for k in stat_keys:
        aggregate[k] += int(stats.get(k, 0))

missing_ids = []
with sft_csv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        qid = str(row.get("question_id", "")).strip()
        qidx = str(row.get("question_index", "")).strip()
        p1 = save_dir / f"{qid}.txt"
        p2 = save_dir / f"{qidx}.txt"
        if not p1.exists() and not p2.exists():
            if qid:
                missing_ids.append(qid)

missing_file.parent.mkdir(parents=True, exist_ok=True)
missing_file.write_text("".join([f"{x}\n" for x in missing_ids]), encoding="utf-8")

summary = {
    "pass_id": pass_id,
    "report_count": len(reports),
    "aggregate_stats": aggregate,
    "missing_count_after_pass": len(missing_ids),
    "missing_ids_path": str(missing_file),
}
with summary_file.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(len(missing_ids))
PY
}

pass_reached=0
missing_count=-1

for pass_id in $(seq 1 "${MAX_PASSES}"); do
  pass_reached="${pass_id}"
  run_one_pass "${pass_id}"
  missing_count="$(summarize_missing "${pass_id}" | tail -n 1 | tr -d '[:space:]')"

  if [[ -z "${missing_count}" ]]; then
    echo "[ERROR] Failed to compute missing count after pass ${pass_id}"
    exit 1
  fi

  echo "[INFO] pass=${pass_id} missing_after_pass=${missing_count}"
  if [[ "${missing_count}" -eq 0 ]]; then
    echo "[PASS] Distillation coverage reached 100% after pass ${pass_id}"
    break
  fi

  if [[ "${pass_id}" -lt "${MAX_PASSES}" ]]; then
    sleep "${PASS_SLEEP_S}"
  fi
done

last_summary="${PASS_REPORT_DIR}/pass_${pass_reached}_summary.json"
last_missing="${PASS_REPORT_DIR}/pass_${pass_reached}_missing_ids.txt"

"${PY_CMD[@]}" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

last_summary = Path(r"${last_summary}")
final_report = Path(r"${FINAL_REPORT_JSON}")
final_failed = Path(r"${FINAL_FAILED_IDS_PATH}")
pass_dir = Path(r"${PASS_REPORT_DIR}")

summary = {}
if last_summary.exists():
    with last_summary.open("r", encoding="utf-8") as f:
        summary = json.load(f)

summary.update(
    {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sft_csv": r"${SFT_CSV}",
        "save_dir": r"${SAVE_DIR}",
        "endpoint": r"${ENDPOINT}",
        "model": r"${DISTILL_MODEL}",
        "num_shards": int("${NUM_SHARDS}"),
        "max_passes": int("${MAX_PASSES}"),
        "passes_run": int("${pass_reached}"),
        "pass_report_dir": str(pass_dir),
    }
)

final_report.parent.mkdir(parents=True, exist_ok=True)
with final_report.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

missing_src = Path(r"${last_missing}")
final_failed.parent.mkdir(parents=True, exist_ok=True)
if missing_src.exists():
    final_failed.write_text(missing_src.read_text(encoding="utf-8"), encoding="utf-8")
else:
    final_failed.write_text("", encoding="utf-8")

print(f"[INFO] wrote final report: {final_report}")
print(f"[INFO] wrote failed ids: {final_failed}")
PY

if [[ "${missing_count}" -ne 0 ]]; then
  echo "[ERROR] Distillation incomplete after ${MAX_PASSES} passes. missing=${missing_count}"
  exit 2
fi

echo "[PASS] Stage C C5 distillation completed."
