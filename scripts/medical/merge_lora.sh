#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <base_model_path> <lora_model_path> <merged_model_save_path>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL_PATH="$1"
LORA_MODEL_PATH="$2"
MERGED_MODEL_SAVE_PATH="$3"

python3 "${SCRIPT_DIR}/merge_lora.py" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --lora_model_path "${LORA_MODEL_PATH}" \
  --merged_model_save_path "${MERGED_MODEL_SAVE_PATH}"
