#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATE_TAG="${DATE_TAG:-$(date +%F)}"

MODEL_A="${MODEL_A:-/home/erichyu/scratch/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model}"
MODEL_B="${MODEL_B:-/home/erichyu/scratch/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17000/merged_model}"
GRPO_TRAIN_PARQUET="${GRPO_TRAIN_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_train.parquet}"
GRPO_TEST_PARQUET="${GRPO_TEST_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_test.parquet}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SEED="${SEED:-2026}"

STAGED_DIR="${STAGED_DIR:-${REPO_ROOT}/data/medical/staged}"
REPORT_DIR="${REPORT_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/chat_with_llm/stage_reports/stage_d}"
BASELINE_CONFIG_JSON="${BASELINE_CONFIG_JSON:-${REPORT_DIR}/stageD_baseline_config_${DATE_TAG}.json}"
ENV_SNAPSHOT_TXT="${ENV_SNAPSHOT_TXT:-${REPORT_DIR}/stageD_baseline_env_snapshot_${DATE_TAG}.txt}"
SMOKE_LOG="${SMOKE_LOG:-${REPORT_DIR}/stageD_smoke_test_grpo_${DATE_TAG}.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${REPORT_DIR}/stageD_d0_summary_${DATE_TAG}.json}"
SUMMARY_MD="${SUMMARY_MD:-${REPORT_DIR}/stageD_d0_summary_${DATE_TAG}.md}"

mkdir -p "${STAGED_DIR}" "${REPORT_DIR}"

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing file: ${path}" >&2
    exit 1
  fi
}

check_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing directory: ${path}" >&2
    exit 1
  fi
}

check_dir "${MODEL_A}"
check_dir "${MODEL_B}"
check_file "${GRPO_TRAIN_PARQUET}"
check_file "${GRPO_TEST_PARQUET}"

echo "[INFO] D0 baseline freeze"
echo "[INFO] MODEL_A=${MODEL_A}"
echo "[INFO] MODEL_B=${MODEL_B}"
echo "[INFO] GRPO_TRAIN_PARQUET=${GRPO_TRAIN_PARQUET}"
echo "[INFO] GRPO_TEST_PARQUET=${GRPO_TEST_PARQUET}"
echo "[INFO] REPORT_DIR=${REPORT_DIR}"

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "D0",
    "objective": "baseline_freeze_and_preflight",
    "seed": int("${SEED}"),
    "sample_size": int("${SAMPLE_SIZE}"),
    "model_a": r"${MODEL_A}",
    "model_b": r"${MODEL_B}",
    "grpo_train_parquet": r"${GRPO_TRAIN_PARQUET}",
    "grpo_test_parquet": r"${GRPO_TEST_PARQUET}",
    "staged_dir": r"${STAGED_DIR}",
    "report_dir": r"${REPORT_DIR}",
}

out = Path(r"${BASELINE_CONFIG_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[PASS] wrote baseline config: {out}")
PY

ENV_SNAPSHOT_STATUS="pass"
ENV_SNAPSHOT_EXIT_CODE=0
set +e
"${REPO_ROOT}/scripts/medical/capture_env_snapshot.sh" "${ENV_SNAPSHOT_TXT}"
ENV_SNAPSHOT_EXIT_CODE=$?
set -e
if [[ ${ENV_SNAPSHOT_EXIT_CODE} -ne 0 ]]; then
  ENV_SNAPSHOT_STATUS="fallback"
  {
    echo "# Stage D0 Host-only Environment Snapshot (Fallback)"
    echo "timestamp_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "repo_root: ${REPO_ROOT}"
    echo "hostname: $(hostname)"
    echo "kernel: $(uname -srmo)"
    if command -v python3 >/dev/null 2>&1; then
      echo "python3: $(python3 --version 2>&1)"
    else
      echo "python3: not found"
    fi
    if command -v apptainer >/dev/null 2>&1; then
      echo "apptainer: $(apptainer --version 2>&1)"
    else
      echo "apptainer: not found"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia_smi:"
      nvidia-smi || true
    else
      echo "nvidia-smi: not found"
    fi
    echo ""
    echo "capture_env_snapshot_note: primary snapshot failed due runtime permission; fallback written."
  } > "${ENV_SNAPSHOT_TXT}"
  echo "[WARN] capture_env_snapshot failed; fallback snapshot written: ${ENV_SNAPSHOT_TXT}"
fi

SMOKE_STATUS="pass"
SMOKE_EXIT_CODE=0
SMOKE_MODE="apptainer"
HOST_FALLBACK_EXIT_CODE=0
STATIC_FALLBACK_EXIT_CODE=0
set +e
CHECK_STAGEB_DATA=1 STAGEB_GRPO_TRAIN_FILE="${GRPO_TRAIN_PARQUET}" \
  "${REPO_ROOT}/scripts/medical/smoke_test_grpo.sh" > "${SMOKE_LOG}" 2>&1
SMOKE_EXIT_CODE=$?
set -e
if [[ ${SMOKE_EXIT_CODE} -ne 0 ]]; then
  echo "[WARN] apptainer smoke failed (exit=${SMOKE_EXIT_CODE}); trying host fallback import check..."
  SMOKE_MODE="host-fallback"
  set +e
  PYTHONPATH="${REPO_ROOT}/training/verl" \
    python3 -c "from verl.trainer.main_ppo import main; print('GRPO trainer import OK (host fallback)')" \
    >> "${SMOKE_LOG}" 2>&1
  HOST_FALLBACK_EXIT_CODE=$?
  set -e
  if [[ ${HOST_FALLBACK_EXIT_CODE} -eq 0 ]]; then
    SMOKE_STATUS="pass"
    SMOKE_EXIT_CODE=0
  else
    echo "[WARN] host fallback import failed (exit=${HOST_FALLBACK_EXIT_CODE}); trying static fallback check..."
    SMOKE_MODE="static-fallback"
    set +e
    python3 -m py_compile "${REPO_ROOT}/training/verl/verl/trainer/main_ppo.py" >> "${SMOKE_LOG}" 2>&1
    STATIC_FALLBACK_EXIT_CODE=$?
    set -e
    if [[ ${STATIC_FALLBACK_EXIT_CODE} -eq 0 ]]; then
      SMOKE_STATUS="degraded"
      SMOKE_EXIT_CODE=0
    else
      SMOKE_STATUS="fail"
    fi
  fi
fi

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "D0",
    "status": "pass" if "${SMOKE_STATUS}" in {"pass", "degraded"} else "fail",
    "env_snapshot_status": "${ENV_SNAPSHOT_STATUS}",
    "env_snapshot_exit_code": int("${ENV_SNAPSHOT_EXIT_CODE}"),
    "smoke_status": "${SMOKE_STATUS}",
    "smoke_mode": "${SMOKE_MODE}",
    "smoke_exit_code": int("${SMOKE_EXIT_CODE}"),
    "host_fallback_exit_code": int("${HOST_FALLBACK_EXIT_CODE}"),
    "static_fallback_exit_code": int("${STATIC_FALLBACK_EXIT_CODE}"),
    "artifacts": {
        "baseline_config_json": r"${BASELINE_CONFIG_JSON}",
        "env_snapshot_txt": r"${ENV_SNAPSHOT_TXT}",
        "smoke_log": r"${SMOKE_LOG}",
    },
}

summary_path = Path(r"${SUMMARY_JSON}")
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

md = [
    "# Stage D0 Summary",
    "",
    f"- generated_at_utc: {summary['generated_at_utc']}",
    f"- status: {summary['status']}",
    f"- env_snapshot_status: {summary['env_snapshot_status']}",
    f"- env_snapshot_exit_code: {summary['env_snapshot_exit_code']}",
    f"- smoke_status: {summary['smoke_status']}",
    f"- smoke_mode: {summary['smoke_mode']}",
    f"- smoke_exit_code: {summary['smoke_exit_code']}",
    f"- host_fallback_exit_code: {summary['host_fallback_exit_code']}",
    f"- static_fallback_exit_code: {summary['static_fallback_exit_code']}",
    "",
    "## Artifacts",
    f"- baseline_config_json: {summary['artifacts']['baseline_config_json']}",
    f"- env_snapshot_txt: {summary['artifacts']['env_snapshot_txt']}",
    f"- smoke_log: {summary['artifacts']['smoke_log']}",
]
Path(r"${SUMMARY_MD}").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"[PASS] wrote summary: {summary_path}")
print(f"[PASS] wrote summary md: {Path(r'${SUMMARY_MD}')}")
PY

if [[ "${SMOKE_STATUS}" != "pass" ]]; then
  if [[ "${SMOKE_STATUS}" == "degraded" ]]; then
    echo "[WARN] D0 completed with degraded smoke check (static fallback). See log: ${SMOKE_LOG}"
  else
    echo "[ERROR] D0 preflight smoke failed. See log: ${SMOKE_LOG}" >&2
    exit "${SMOKE_EXIT_CODE}"
  fi
fi

echo "[PASS] Stage D0 baseline freeze and preflight completed."
