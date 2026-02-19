#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Host: $(hostname)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found"
  exit 1
fi

echo "[INFO] GPU listing"
nvidia-smi -L

if nvidia-smi -L | grep -q "MIG"; then
  echo "[WARN] MIG devices detected. This is not full-GPU mode for 235B teacher deployment."
  exit 2
fi

GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
echo "[INFO] Full GPU count: ${GPU_COUNT}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[ERROR] No full GPUs detected"
  exit 1
fi

echo "[PASS] Full GPU environment check passed"
