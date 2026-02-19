#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  echo "[INFO] Stage C debug loop attempt ${attempt}/${MAX_ATTEMPTS}"
  if bash "${SCRIPT_DIR}/run_stagec_pipeline_smoke_0p6b.sh"; then
    echo "[PASS] debug loop succeeded at attempt ${attempt}"
    exit 0
  fi
  echo "[WARN] attempt ${attempt} failed"
  if [[ "${attempt}" -lt "${MAX_ATTEMPTS}" ]]; then
    echo "[INFO] sleep ${SLEEP_SECONDS}s before retry"
    sleep "${SLEEP_SECONDS}"
  fi
done

echo "[ERROR] debug loop exhausted attempts"
exit 1
