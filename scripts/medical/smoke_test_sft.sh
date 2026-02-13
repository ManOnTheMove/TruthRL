#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
  echo "[ERROR] apptainer not found in PATH" >&2
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "[ERROR] Apptainer image not found: ${IMAGE_PATH}" >&2
  echo "Set APPTAINER_IMAGE to your truthrl.sif path." >&2
  exit 1
fi

PYTHONPATH_IN_CONTAINER="${REPO_ROOT}/training/verl"
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

NV_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  NV_FLAG=(--nv)
fi

echo "[INFO] Running SFT import smoke test in Apptainer"
echo "[INFO] IMAGE_PATH=${IMAGE_PATH}"
echo "[INFO] PYTHONPATH=${PYTHONPATH_IN_CONTAINER}"
echo "[INFO] APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo "[INFO] APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"

apptainer exec "${NV_FLAG[@]}" --cleanenv \
  --env "PYTHONPATH=${PYTHONPATH_IN_CONTAINER}" \
  "${IMAGE_PATH}" \
  python3 -c "from verl.trainer.fsdp_sft_trainer import main; print('SFT trainer import OK')"

echo "[PASS] SFT smoke test succeeded"
