#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/medical}"
PID_FILE="${PID_FILE:-${LOG_DIR}/qwen235b_server.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[INFO] PID file not found: ${PID_FILE}"
  exit 0
fi

pid="$(cat "${PID_FILE}" || true)"
if [[ -z "${pid}" ]]; then
  echo "[WARN] empty pid file"
  rm -f "${PID_FILE}"
  exit 0
fi

if kill -0 "${pid}" >/dev/null 2>&1; then
  echo "[INFO] stopping qwen235b server pid=${pid}"
  kill "${pid}" || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[WARN] force killing pid=${pid}"
    kill -9 "${pid}" || true
  fi
else
  echo "[INFO] process already stopped pid=${pid}"
fi

rm -f "${PID_FILE}"
echo "[PASS] qwen235b server stopped"
