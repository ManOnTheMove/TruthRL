#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

is_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

print_command() {
  printf "[DRY-RUN] Command:"
  printf " %q" "$@"
  printf "\n"
}

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_train_with_ook_d26.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/data/medical/verl/medqa_grpo_test.parquet}"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/data/medical/staged/d3_hard_ook_truthrl}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${LOG_DIR}/hydra/${SLURM_JOB_ID:-manual}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-/tmp/d3_${JOB_ID}}"
RAY_TMPDIR="${RAY_TMPDIR:-${LOCAL_TMP_ROOT}/ray}"

USE_APPTAINER="${USE_APPTAINER:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ROLLOUT_N="${ROLLOUT_N:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
if (( ROLLOUT_MAX_BATCHED_TOKENS < ROLLOUT_MAX_MODEL_LEN )); then
  echo "[WARN] ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS} is smaller than ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN}; raising it to ${ROLLOUT_MAX_MODEL_LEN}." >&2
  ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_MODEL_LEN}"
fi
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-50}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
LR="${LR:-1e-6}"
K="${K:-1}"
PROJECT_NAME="${PROJECT_NAME:-stage-d-d3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-truthrl-hard-ook-d3}"

if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
  mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${HYDRA_RUN_DIR}" "${RAY_TMPDIR}" "${LOCAL_TMP_ROOT}"
fi
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_TMPDIR
export TMPDIR="${D3_TMPDIR:-${LOCAL_TMP_ROOT}/tmp}"
if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
  mkdir -p "${TMPDIR}"
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] MODEL_PATH invalid: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "[ERROR] Missing TRAIN_PARQUET: ${TRAIN_PARQUET}" >&2
  exit 1
fi
if [[ ! -f "${VAL_PARQUET}" ]]; then
  echo "[ERROR] Missing VAL_PARQUET: ${VAL_PARQUET}" >&2
  exit 1
fi

if [[ "${USE_APPTAINER}" == "1" ]]; then
  if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    IMAGE_PATH="${APPTAINER_IMAGE}"
  elif [[ -f "${REPO_ROOT}/truthrl.sif" ]]; then
    IMAGE_PATH="${REPO_ROOT}/truthrl.sif"
  else
    IMAGE_PATH="$(cd "${REPO_ROOT}/.." && pwd)/envs/truthrl_apptainer/truthrl.sif"
  fi

  if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
    if ! command -v apptainer >/dev/null 2>&1; then
      if command -v module >/dev/null 2>&1; then
        module load apptainer >/dev/null 2>&1 || true
      fi
    fi
    if ! command -v apptainer >/dev/null 2>&1; then
      echo "[ERROR] apptainer not found" >&2
      exit 1
    fi
    if [[ ! -f "${IMAGE_PATH}" ]]; then
      echo "[ERROR] Apptainer image not found: ${IMAGE_PATH}" >&2
      exit 1
    fi
  fi

  NODE_TMPDIR="${SLURM_TMPDIR:-/tmp}"
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${NODE_TMPDIR}/apptainer_cache_${USER}}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${NODE_TMPDIR}/apptainer_tmp_${USER}}"
  if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
    mkdir -p "${NODE_TMPDIR}" >/dev/null 2>&1 || true
    mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
  fi

  SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
  CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/.cache}"
  CONFIG_ROOT="${CONFIG_ROOT:-${SCRATCH_ROOT}/.config}"
  if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
    mkdir -p "${CACHE_ROOT}/huggingface/modules" "${CACHE_ROOT}/flashinfer" "${CONFIG_ROOT}/vllm"
  fi

  export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CONFIG_ROOT}}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
  export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CACHE_ROOT}/cuda}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
  export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${CACHE_ROOT}/flashinfer}"
  if ! is_truthy "${DRY_RUN}" && ! is_truthy "${PREFLIGHT_ONLY}"; then
    mkdir -p "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"
  fi
  if [[ -z "${APPTAINER_BINDPATH:-}" ]]; then
    export APPTAINER_BINDPATH="/project:/project,/home/${USER}:/home/${USER},/scratch:/scratch,${CACHE_ROOT}:/home/${USER}/.cache,${CONFIG_ROOT}:/home/${USER}/.config"
  fi

  NV_FLAG=()
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG=(--nv)
  fi

  BIND_FLAG=()
  if [[ -n "${APPTAINER_BINDPATH:-}" ]]; then
    BIND_FLAG=(--bind "${APPTAINER_BINDPATH}")
  fi

  CONTAINER_ENVS="PYTHONPATH=${REPO_ROOT}/training/verl,RAY_TMPDIR=${RAY_TMPDIR},TMPDIR=${TMPDIR},XDG_CONFIG_HOME=/home/${USER}/.config,XDG_CACHE_HOME=/home/${USER}/.cache,HF_HOME=/home/${USER}/.cache/huggingface,HF_MODULES_CACHE=/home/${USER}/.cache/huggingface/modules,TRITON_CACHE_DIR=/home/${USER}/.cache/triton,CUDA_CACHE_PATH=/home/${USER}/.cache/cuda,TORCHINDUCTOR_CACHE_DIR=/home/${USER}/.cache/torchinductor,FLASHINFER_CACHE_DIR=/home/${USER}/.cache/flashinfer"
  PY_CMD=(apptainer exec "${NV_FLAG[@]}" "${BIND_FLAG[@]}" --cleanenv --env "${CONTAINER_ENVS}" "${IMAGE_PATH}" python3)
else
  PY_CMD=(python3)
fi

echo "[INFO] D3 hard-OOK GRPO run"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TRAIN_PARQUET=${TRAIN_PARQUET}"
echo "[INFO] VAL_PARQUET=${VAL_PARQUET}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "[INFO] RAY_TMPDIR=${RAY_TMPDIR}"
echo "[INFO] TMPDIR=${TMPDIR}"
echo "[INFO] VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}"
echo "[INFO] ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN}"
echo "[INFO] ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS}"
echo "[INFO] K=${K}"

TRAIN_CMD=("${PY_CMD[@]}" -m verl.trainer.main_ppo \
  --config-name _generated_ppo_trainer \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=naive \
  custom_reward_function.path="${REPO_ROOT}/training/verl/verl/utils/reward_score/clinical_medqa_reward.py" \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.stage_mode=d3_hard_ook \
  +custom_reward_function.reward_kwargs.k="${K}" \
  +custom_reward_function.reward_kwargs.enable_format=False \
  +custom_reward_function.reward_kwargs.enable_consistency=False \
  algorithm.use_kl_in_reward=False \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" "$@")

if is_truthy "${DRY_RUN}" || is_truthy "${PREFLIGHT_ONLY}"; then
  print_command "${TRAIN_CMD[@]}"
  echo "[DRY-RUN] D3 hard-OOK command was not executed."
  exit 0
fi

"${TRAIN_CMD[@]}"

echo "[PASS] D3 hard-OOK run completed."
