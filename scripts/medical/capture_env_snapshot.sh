#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_FILE="${1:-${REPO_ROOT}/env_snapshot.txt}"
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

NODE_TMPDIR="${SLURM_TMPDIR:-/tmp}"
if ! mkdir -p "${NODE_TMPDIR}" >/dev/null 2>&1; then
  NODE_TMPDIR="/tmp"
fi
CACHE_CANDIDATE="${APPTAINER_CACHEDIR:-${NODE_TMPDIR}/apptainer_cache_${USER}}"
TMP_CANDIDATE="${APPTAINER_TMPDIR:-${NODE_TMPDIR}/apptainer_tmp_${USER}}"

if ! mkdir -p "${CACHE_CANDIDATE}" >/dev/null 2>&1; then
  CACHE_CANDIDATE="/tmp/apptainer_cache_${USER}"
  mkdir -p "${CACHE_CANDIDATE}"
fi
if ! mkdir -p "${TMP_CANDIDATE}" >/dev/null 2>&1; then
  TMP_CANDIDATE="/tmp/apptainer_tmp_${USER}"
  mkdir -p "${TMP_CANDIDATE}"
fi
export APPTAINER_CACHEDIR="${CACHE_CANDIDATE}"
export APPTAINER_TMPDIR="${TMP_CANDIDATE}"

{
  echo "# TruthRL Environment Snapshot"
  echo "timestamp_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "repo_root: ${REPO_ROOT}"
  echo "apptainer_image: ${IMAGE_PATH}"
  echo "git_branch: $(git -C "${REPO_ROOT}" branch --show-current)"
  echo "git_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo ""

  echo "## Host"
  echo "hostname: $(hostname)"
  echo "kernel: $(uname -srmo)"
  echo ""

  echo "## Tooling"
  if command -v python3 >/dev/null 2>&1; then
    echo "python3: $(python3 --version 2>&1)"
  else
    echo "python3: not found"
  fi

  if command -v pip >/dev/null 2>&1; then
    echo "pip: $(pip --version 2>&1)"
  else
    echo "pip: not found"
  fi

  if command -v apptainer >/dev/null 2>&1; then
    echo "apptainer: $(apptainer --version 2>&1)"
  else
    echo "apptainer: not found"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "## NVIDIA"
    if nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv,noheader 2>/dev/null; then
      :
    else
      echo "nvidia-smi present but GPU info unavailable"
    fi
  else
    echo "nvidia-smi: not found"
  fi

  echo ""
  echo "## Python Packages (top-level hints)"
  python3 - <<'PY'
import importlib
for pkg in ["torch", "vllm", "ray", "transformers", "flash_attn"]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "unknown")
        print(f"{pkg}: {v}")
    except Exception as e:
        print(f"{pkg}: unavailable ({type(e).__name__})")
PY

  echo ""
  echo "## Container (cleanenv)"
  if command -v apptainer >/dev/null 2>&1 && [[ -f "${IMAGE_PATH}" ]]; then
    apptainer exec --cleanenv \
      --env "PYTHONPATH=${REPO_ROOT}/training/verl" \
      "${IMAGE_PATH}" \
      python3 - <<'PY'
import importlib
import platform
import subprocess
import sys

print(f"python: {sys.version.split()[0]}")
print(f"platform: {platform.platform()}")
for pkg in ["verl", "torch", "vllm", "ray", "transformers", "flash_attn"]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "unknown")
        print(f"{pkg}: {v}")
    except Exception as e:
        print(f"{pkg}: unavailable ({type(e).__name__})")

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    ).strip()
    print(f"cuda_driver_info: {out if out else 'unavailable'}")
except Exception:
    print("cuda_driver_info: unavailable")
PY
  else
    echo "container_snapshot: skipped (apptainer or image missing)"
  fi
} > "${OUT_FILE}"

echo "[PASS] Wrote environment snapshot to ${OUT_FILE}"
