#!/usr/bin/env bash
set -euo pipefail

OUTPUT_LOG="${1:-}"
SAMPLE_INTERVAL_S="${2:-5}"
WINDOW_S="${3:-60}"

if [[ -z "${OUTPUT_LOG}" ]]; then
  echo "Usage: $0 <output_log> [sample_interval_s=5] [window_s=60]"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_LOG}")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found"
  exit 1
fi

mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits)
if [[ "${#GPU_LINES[@]}" -eq 0 ]]; then
  echo "[ERROR] no visible GPUs"
  exit 1
fi

declare -a GPU_IDS=()
declare -A MEM_TOTAL=()
declare -A UTIL_SUM=()
declare -A MEM_SUM=()
declare -A SAMPLE_CNT=()

for line in "${GPU_LINES[@]}"; do
  idx="$(echo "${line}" | cut -d',' -f1 | xargs)"
  total="$(echo "${line}" | cut -d',' -f2 | xargs)"
  GPU_IDS+=("${idx}")
  MEM_TOTAL["${idx}"]="${total}"
  UTIL_SUM["${idx}"]=0
  MEM_SUM["${idx}"]=0
  SAMPLE_CNT["${idx}"]=0
done

window_start="$(date +%s)"
echo "[INFO] GPU minute monitor started: output=${OUTPUT_LOG} interval=${SAMPLE_INTERVAL_S}s window=${WINDOW_S}s" | tee -a "${OUTPUT_LOG}"

while true; do
  while IFS=',' read -r idx util mem _; do
    idx="$(echo "${idx}" | xargs)"
    util="$(echo "${util}" | xargs)"
    mem="$(echo "${mem}" | xargs)"

    if [[ ! "${util}" =~ ^[0-9]+$ ]]; then
      util=0
    fi
    if [[ ! "${mem}" =~ ^[0-9]+$ ]]; then
      mem=0
    fi

    UTIL_SUM["${idx}"]=$(( UTIL_SUM["${idx}"] + util ))
    MEM_SUM["${idx}"]=$(( MEM_SUM["${idx}"] + mem ))
    SAMPLE_CNT["${idx}"]=$(( SAMPLE_CNT["${idx}"] + 1 ))
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)

  now="$(date +%s)"
  elapsed=$(( now - window_start ))
  if (( elapsed >= WINDOW_S )); then
    ts="$(date '+%F %T')"
    {
      echo "[${ts}] 1-minute GPU average"
      for idx in "${GPU_IDS[@]}"; do
        cnt="${SAMPLE_CNT[${idx}]}"
        total="${MEM_TOTAL[${idx}]}"
        if (( cnt > 0 )); then
          util_avg="$(awk "BEGIN{printf \"%.2f\", ${UTIL_SUM[${idx}]}/${cnt}}")"
          mem_avg="$(awk "BEGIN{printf \"%.2f\", ${MEM_SUM[${idx}]}/${cnt}}")"
          mem_pct="$(awk "BEGIN{printf \"%.2f\", (${MEM_SUM[${idx}]}/${cnt})*100/${total}}")"
        else
          util_avg="0.00"
          mem_avg="0.00"
          mem_pct="0.00"
        fi
        echo "  GPU${idx}: util_avg=${util_avg}% mem_avg=${mem_avg}MiB/${total}MiB (${mem_pct}%) samples=${cnt}"
        UTIL_SUM["${idx}"]=0
        MEM_SUM["${idx}"]=0
        SAMPLE_CNT["${idx}"]=0
      done
    } | tee -a "${OUTPUT_LOG}"
    window_start="${now}"
  fi

  sleep "${SAMPLE_INTERVAL_S}"
done

