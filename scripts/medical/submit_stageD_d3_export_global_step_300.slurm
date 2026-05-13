#!/usr/bin/env bash
#SBATCH --job-name=stageD-d3-export-gs300
#SBATCH --account=def-zshakeri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/erichyu/stageD_d3_hard_ook/export/logs/stageD-d3-export-gs300-%j.out
#SBATCH --error=/scratch/erichyu/stageD_d3_hard_ook/export/logs/stageD-d3-export-gs300-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/project/def-zshakeri/erichyu/embc/TruthRL}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/def-zshakeri/erichyu/embc}"
LOCAL_DIR="${LOCAL_DIR:-/scratch/erichyu/stageD_d3_hard_ook/run/checkpoints/global_step_300/actor}"
TARGET_DIR="${TARGET_DIR:-/scratch/erichyu/stageD_d3_hard_ook/export/global_step_300_hf}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/envs/truthrl_apptainer/truthrl.sif}"

echo "[INFO] Stage D D3 checkpoint export"
echo "[INFO] HOST=$(hostname)"
echo "[INFO] DATE=$(date -Is)"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] LOCAL_DIR=${LOCAL_DIR}"
echo "[INFO] TARGET_DIR=${TARGET_DIR}"
echo "[INFO] APPTAINER_IMAGE=${APPTAINER_IMAGE}"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "[ERROR] Missing LOCAL_DIR: ${LOCAL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${LOCAL_DIR}/fsdp_config.json" ]]; then
  echo "[ERROR] LOCAL_DIR does not look like a verl FSDP actor checkpoint: ${LOCAL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "[ERROR] Missing APPTAINER_IMAGE: ${APPTAINER_IMAGE}" >&2
  exit 1
fi

if [[ -e "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] TARGET_DIR already exists and is non-empty: ${TARGET_DIR}" >&2
  echo "[ERROR] Refusing to overwrite existing export output." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

if ! command -v apptainer >/dev/null 2>&1; then
  module load apptainer >/dev/null 2>&1 || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[ERROR] apptainer not found" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/training/verl"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-192}"

apptainer exec \
  --bind /project:/project,/scratch:/scratch,/home/${USER}:/home/${USER} \
  --cleanenv \
  --env "PYTHONPATH=${PYTHONPATH},OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  "${APPTAINER_IMAGE}" \
  python3 -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"

echo "[INFO] Export directory contents:"
find "${TARGET_DIR}" -maxdepth 2 -type f | sort

if [[ ! -f "${TARGET_DIR}/config.json" ]]; then
  echo "[ERROR] Missing exported config.json" >&2
  exit 1
fi

if [[ ! -f "${TARGET_DIR}/tokenizer.json" ]]; then
  echo "[ERROR] Missing exported tokenizer.json" >&2
  exit 1
fi

if ! find "${TARGET_DIR}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.bin' \) -print -quit | grep -q .; then
  echo "[ERROR] Export finished but no model weight file was found in ${TARGET_DIR}" >&2
  exit 1
fi

echo "[PASS] D3 global_step_300 FSDP checkpoint exported to HF-style directory."
