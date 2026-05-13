#!/usr/bin/env bash
#SBATCH --job-name=stageD-phase23-eval
#SBATCH --account=def-zshakeri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/logs/stageD-phase23-eval-%j.out
#SBATCH --error=/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/logs/stageD-phase23-eval-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-zshakeri/erichyu/embc}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/TruthRL}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/envs/truthrl_apptainer/truthrl.sif}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3}"
RUN_ID="${RUN_ID:-phase23_20260511_standard_d26_stagec_vs_d3}"

STAGEC_MODEL="${STAGEC_MODEL:-${PROJECT_ROOT}/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model}"
D3_MODEL="${D3_MODEL:-/scratch/erichyu/stageD_d3_hard_ook/export/global_step_300_hf}"

STANDARD_PARQUET="${STANDARD_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_test.parquet}"
D26_PARQUET="${D26_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_train_with_ook_d26.parquet}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

echo "[INFO] Stage D phase 2-3 eval"
echo "[INFO] HOST=$(hostname)"
echo "[INFO] DATE=$(date -Is)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] STAGEC_MODEL=${STAGEC_MODEL}"
echo "[INFO] D3_MODEL=${D3_MODEL}"

for path in "${APPTAINER_IMAGE}" "${STAGEC_MODEL}/config.json" "${D3_MODEL}/config.json" "${STANDARD_PARQUET}" "${D26_PARQUET}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/logs"

NODE_TMPDIR="${SLURM_TMPDIR:-/tmp}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/.cache}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCRATCH_ROOT}/.config}"
mkdir -p "${CACHE_ROOT}/huggingface/modules" "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/cuda" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/flashinfer" "${CONFIG_ROOT}/vllm"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${NODE_TMPDIR}/apptainer_cache_${USER}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${NODE_TMPDIR}/apptainer_tmp_${USER}}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CONFIG_ROOT}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CACHE_ROOT}/cuda}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${CACHE_ROOT}/flashinfer}"
export PYTHONPATH="${REPO_ROOT}/training/verl"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

CONTAINER_ENVS="PYTHONPATH=${PYTHONPATH},XDG_CONFIG_HOME=/home/${USER}/.config,XDG_CACHE_HOME=/home/${USER}/.cache,HF_HOME=/home/${USER}/.cache/huggingface,HF_MODULES_CACHE=/home/${USER}/.cache/huggingface/modules,TRITON_CACHE_DIR=/home/${USER}/.cache/triton,CUDA_CACHE_PATH=/home/${USER}/.cache/cuda,TORCHINDUCTOR_CACHE_DIR=/home/${USER}/.cache/torchinductor,FLASHINFER_CACHE_DIR=/home/${USER}/.cache/flashinfer,OMP_NUM_THREADS=${OMP_NUM_THREADS}"

PY_CMD=(
  apptainer exec
  --nv
  --bind "/project:/project,/scratch:/scratch,/home/${USER}:/home/${USER},${CACHE_ROOT}:/home/${USER}/.cache,${CONFIG_ROOT}:/home/${USER}/.config"
  --cleanenv
  --env "${CONTAINER_ENVS}"
  "${APPTAINER_IMAGE}"
  python3
)

"${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/stageD_eval_collect.py" \
  --model-path "${STAGEC_MODEL}" \
  --model-id stagec_base \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --standard-parquet "${STANDARD_PARQUET}" \
  --d26-parquet "${D26_PARQUET}" \
  --groups standard_medqa,d26_hard_ook \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --overwrite

"${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/stageD_eval_collect.py" \
  --model-path "${D3_MODEL}" \
  --model-id d3_global_step_300 \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --standard-parquet "${STANDARD_PARQUET}" \
  --d26-parquet "${D26_PARQUET}" \
  --groups standard_medqa,d26_hard_ook \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --overwrite

"${PY_CMD[@]}" "${REPO_ROOT}/scripts/medical/stageD_eval_compare.py" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --model-ids stagec_base d3_global_step_300 \
  --baseline-model-id stagec_base \
  --candidate-model-id d3_global_step_300

echo "[PASS] Stage D phase 2-3 eval completed."
