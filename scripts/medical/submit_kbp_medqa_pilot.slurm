#!/bin/bash
#SBATCH --job-name=stageD-d25-kbp-pilot
#SBATCH --account=def-zshakeri_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --qos=normal
#SBATCH --nice=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/erichyu/projects/def-zshakeri/erichyu/embc"
cd "${PROJECT_ROOT}/TruthRL"

export USE_APPTAINER="${USE_APPTAINER:-1}"
export RUN_ID="${RUN_ID:-kbp_20260301_stagec-c8-8b_medqa_grpo_train_pilot10}"
export SPLIT="${SPLIT:-medqa_grpo_train}"
export QUESTION_LIMIT="${QUESTION_LIMIT:-10}"

export PROBES_PER_QUESTION="${PROBES_PER_QUESTION:-256}"
export N_CHUNK="${N_CHUNK:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"

export MODEL_PATH="${MODEL_PATH:-/home/erichyu/scratch/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model}"
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
export RESUME="${RESUME:-0}"

bash scripts/medical/run_kbp_medqa.sh "$@"
