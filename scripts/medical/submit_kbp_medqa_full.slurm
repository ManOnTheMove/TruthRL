#!/bin/bash
#SBATCH --job-name=stageD-d25-kbp-full
#SBATCH --account=def-zshakeri
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --qos=normal
#SBATCH --nice=0
#SBATCH --output=/scratch/erichyu/kbp_test/%x-%j.out
#SBATCH --error=/scratch/erichyu/kbp_test/%x-%j.err

set -euo pipefail

# Priority: explicit PROJECT_ROOT -> SLURM_SUBMIT_DIR -> script-relative fallback
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/../models" ]] && [[ -d "${SLURM_SUBMIT_DIR}/scripts/medical" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${PROJECT_ROOT}/TruthRL"

SCRATCH_ROOT="/scratch/${USER}"
mkdir -p "${SCRATCH_ROOT}/embc_kbp" \
         "${SCRATCH_ROOT}/.cache/huggingface/modules" \
         "${SCRATCH_ROOT}/.cache/flashinfer" \
         "${SCRATCH_ROOT}/.config/vllm" \
         "/scratch/erichyu/kbp_test"

export USE_APPTAINER="${USE_APPTAINER:-1}"
export RUN_ID="${RUN_ID:-kbp_20260301_stagec-c8-8b_medqa_grpo_train}"
export SPLIT="${SPLIT:-medqa_grpo_train}"
export QUESTION_LIMIT="${QUESTION_LIMIT:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/TruthRL/data/medical/kbp}"
export APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/envs/truthrl_apptainer/truthrl.sif}"
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-/project:/project,/home/${USER}:/home/${USER},/scratch:/scratch,${SCRATCH_ROOT}/.cache:/home/${USER}/.cache,${SCRATCH_ROOT}/.config:/home/${USER}/.config}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${SCRATCH_ROOT}/.config}"

export PROBES_PER_QUESTION="${PROBES_PER_QUESTION:-256}"
export N_CHUNK="${N_CHUNK:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model}"
export MODEL_ID="${MODEL_ID:-stagec-c8-8b-merged}"

export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
export STOP_STR="${STOP_STR:-</answer>}"
export INCLUDE_STOP_STR_IN_OUTPUT="${INCLUDE_STOP_STR_IN_OUTPUT:-1}"
export SEED_BASE="${SEED_BASE:-20260301}"

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export PART_ROWS="${PART_ROWS:-80000}"
export RESUME="${RESUME:-1}"

bash scripts/medical/run_kbp_medqa.sh "$@"
