#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-prod}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/def-zshakeri/erichyu/embc}"
TRUTHRL_ROOT="${PROJECT_ROOT}/TruthRL"
FULL_SCRIPT="${TRUTHRL_ROOT}/scripts/medical/submit_kbp_medqa_full.slurm"
MERGE_SCRIPT="${TRUTHRL_ROOT}/scripts/medical/submit_kbp_medqa_merge.slurm"
ACCOUNT="${ACCOUNT:-def-zshakeri}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/TruthRL/data/medical/kbp}"
SPLIT="${SPLIT:-medqa_grpo_train}"
POSTPROCESS_SUBMIT_MODE="${POSTPROCESS_SUBMIT_MODE:-cpu_login}"

CPU_LOGIN_HOST="${CPU_LOGIN_HOST:-tri-login01}"
CPU_MERGE_PARTITION="${CPU_MERGE_PARTITION:-compute}"
CPU_MERGE_CPUS_PER_TASK="${CPU_MERGE_CPUS_PER_TASK:-32}"
CPU_MERGE_TIME_LIMIT="${CPU_MERGE_TIME_LIMIT:-04:00:00}"
WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-30}"
WATCH_DIR="${WATCH_DIR:-/scratch/${USER}/kbp_test}"

# Legacy GPU-side postprocess mode (kept as fallback)
MERGE_PARTITION="${MERGE_PARTITION:-compute_full_node}"
MERGE_GPUS_PER_NODE="${MERGE_GPUS_PER_NODE:-4}"
GUARD_PARTITION="${GUARD_PARTITION:-compute_full_node}"
GUARD_GPUS_PER_NODE="${GUARD_GPUS_PER_NODE:-4}"
MERGE_TIME_LIMIT="${MERGE_TIME_LIMIT:-08:00:00}"
GUARD_TIME_LIMIT="${GUARD_TIME_LIMIT:-00:30:00}"

mkdir -p "${OUTPUT_ROOT}/runs" /scratch/erichyu/kbp_test "${WATCH_DIR}"

if [[ ! -f "${FULL_SCRIPT}" ]]; then
  echo "[ERROR] missing full submit script: ${FULL_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "[ERROR] missing merge submit script: ${MERGE_SCRIPT}" >&2
  exit 1
fi

KILL_INVALID_DEP_FLAG=()
if sbatch --help 2>&1 | grep -q -- '--kill-on-invalid-dep'; then
  KILL_INVALID_DEP_FLAG=(--kill-on-invalid-dep=yes)
fi

if [[ "${MODE}" == "debug" ]]; then
  FINAL_RUN_ID="${FINAL_RUN_ID:-kbp_parallel_debug_$(date +%Y%m%d_%H%M%S)}"
  BASE_RUN_ID="${BASE_RUN_ID:-}"
  SHARDS=("0:2" "2:4" "4:6" "6:8")
  PARTITION="${PARTITION:-debug}"
  TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
  NUM_WORKERS="${NUM_WORKERS:-1}"
  GPU_IDS="${GPU_IDS:-0}"
  PROBES_PER_QUESTION="${PROBES_PER_QUESTION:-16}"
  N_CHUNK="${N_CHUNK:-4}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
  QUESTION_LIMIT_FINAL="${QUESTION_LIMIT_FINAL:-8}"
else
  FINAL_RUN_ID="${FINAL_RUN_ID:-kbp_20260301_stagec-c8-8b_medqa_grpo_train}"
  BASE_RUN_ID="${BASE_RUN_ID:-${FINAL_RUN_ID}_base640k}"
  BASE_SOURCE_OUTPUT_ROOT="${BASE_SOURCE_OUTPUT_ROOT:-${PROJECT_ROOT}/TruthRL/data/medical/kbp}"
  BASE_SOURCE_RUN_ID="${BASE_SOURCE_RUN_ID:-${FINAL_RUN_ID}}"
  SHARDS=("2500:3148" "3148:3795" "3795:4442" "4442:5089")
  PARTITION="${PARTITION:-compute_full_node}"
  TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
  NUM_WORKERS="${NUM_WORKERS:-4}"
  GPU_IDS="${GPU_IDS:-0,1,2,3}"
  PROBES_PER_QUESTION="${PROBES_PER_QUESTION:-256}"
  N_CHUNK="${N_CHUNK:-16}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
  QUESTION_LIMIT_FINAL="${QUESTION_LIMIT_FINAL:-5089}"

  SNAPSHOT_DIR="${OUTPUT_ROOT}/runs/${BASE_RUN_ID}"
  if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
    SRC_DIR="${BASE_SOURCE_OUTPUT_ROOT}/runs/${BASE_SOURCE_RUN_ID}"
    if [[ ! -d "${SRC_DIR}" ]]; then
      echo "[ERROR] base source run not found: ${SRC_DIR}" >&2
      exit 1
    fi
    echo "[INFO] create base snapshot: ${SNAPSHOT_DIR}"
    rsync -a "${SRC_DIR}/" "${SNAPSHOT_DIR}/"
  fi
fi

submit_shard() {
  local shard_name="$1"
  local start_rank="$2"
  local end_rank="$3"
  local run_id="$4"

  env GPU_IDS="${GPU_IDS}" sbatch --parsable \
    --account="${ACCOUNT}" \
    --job-name="${shard_name}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},RUN_ID=${run_id},OUTPUT_ROOT=${OUTPUT_ROOT},SPLIT=${SPLIT},QUESTION_LIMIT=${end_rank},QUESTION_RANK_START=${start_rank},QUESTION_RANK_END=${end_rank},RESUME=0,NUM_WORKERS=${NUM_WORKERS},PROBES_PER_QUESTION=${PROBES_PER_QUESTION},N_CHUNK=${N_CHUNK},MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" \
    "${FULL_SCRIPT}"
}

declare -a SHARD_JOB_IDS=()
declare -a SHARD_RUN_IDS=()
merge_job_id=""
guard_job_id=""

cleanup_on_error() {
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "[ERROR] submission failed, cleaning up submitted jobs" >&2
    if [[ ${#SHARD_JOB_IDS[@]} -gt 0 ]]; then
      scancel "${SHARD_JOB_IDS[@]}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${merge_job_id}" ]]; then
      scancel "${merge_job_id}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${guard_job_id}" ]]; then
      scancel "${guard_job_id}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup_on_error EXIT

idx=1
for item in "${SHARDS[@]}"; do
  start_rank="${item%%:*}"
  end_rank="${item##*:}"
  shard_run_id="${FINAL_RUN_ID}_shard${idx}"
  shard_job_name="stageD-d25-shard${idx}-${start_rank}-${end_rank}"
  jid="$(submit_shard "${shard_job_name}" "${start_rank}" "${end_rank}" "${shard_run_id}")"
  SHARD_JOB_IDS+=("${jid}")
  SHARD_RUN_IDS+=("${shard_run_id}")
  echo "[INFO] shard${idx} job=${jid} run_id=${shard_run_id} range=${start_rank}..$((end_rank-1))"
  idx=$((idx + 1))
done

afterok_ids="$(IFS=:; echo "${SHARD_JOB_IDS[*]}")"
merge_shards="$(IFS=:; echo "${SHARD_RUN_IDS[*]}")"

if [[ "${POSTPROCESS_SUBMIT_MODE}" == "cpu_login" ]]; then
  watcher_script="${WATCH_DIR}/watch_${FINAL_RUN_ID}.sh"
  watcher_log="${WATCH_DIR}/watch_${FINAL_RUN_ID}.log"

  cat > "${watcher_script}" <<WATCHER_EOF
#!/usr/bin/env bash
set -euo pipefail

JOB_IDS=(${SHARD_JOB_IDS[*]})
JOB_ID_CSV="$(IFS=,; echo "${SHARD_JOB_IDS[*]}")"
CPU_LOGIN_HOST="${CPU_LOGIN_HOST}"
ACCOUNT="${ACCOUNT}"
PROJECT_ROOT="${PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"
FINAL_RUN_ID="${FINAL_RUN_ID}"
BASE_RUN_ID="${BASE_RUN_ID}"
MERGE_SCRIPT="${MERGE_SCRIPT}"
SHARD_RUN_IDS="${merge_shards}"
SPLIT="${SPLIT}"
QUESTION_LIMIT_FINAL="${QUESTION_LIMIT_FINAL}"
WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC}"
CPU_MERGE_PARTITION="${CPU_MERGE_PARTITION}"
CPU_MERGE_CPUS_PER_TASK="${CPU_MERGE_CPUS_PER_TASK}"
CPU_MERGE_TIME_LIMIT="${CPU_MERGE_TIME_LIMIT}"

echo "[WATCHER] started at \$(date -Is) host=\$(hostname) run_id=\${FINAL_RUN_ID}"
echo "[WATCHER] shard jobs: \${JOB_IDS[*]}"

is_failed_state() {
  case "\$1" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

while true; do
  declare -A STATE_BY_ID=()

  while read -r jid st; do
    [[ -z "\${jid:-}" ]] && continue
    STATE_BY_ID["\$jid"]="\${st%%+*}"
  done < <(squeue -h -j "\$JOB_ID_CSV" -o "%i %T" 2>/dev/null || true)

  while read -r jid st; do
    [[ -z "\${jid:-}" ]] && continue
    [[ -n "\${STATE_BY_ID[\$jid]:-}" ]] && continue
    STATE_BY_ID["\$jid"]="\${st%%+*}"
  done < <(sacct -n -X -j "\$JOB_ID_CSV" --format=JobIDRaw,State 2>/dev/null | awk '\$1 ~ /^[0-9]+$/ {print \$1, \$2}' || true)

  completed=0
  failed=0
  summary=()

  for jid in "\${JOB_IDS[@]}"; do
    st="\${STATE_BY_ID[\$jid]:-UNKNOWN}"
    summary+=("\${jid}:\${st}")

    if [[ "\$st" == "COMPLETED" ]]; then
      completed=\$((completed + 1))
      continue
    fi

    if is_failed_state "\$st"; then
      failed=\$((failed + 1))
    fi
  done

  echo "[WATCHER] \$(date -Is) states=\${summary[*]}"

  if [[ \$failed -gt 0 ]]; then
    echo "[WATCHER][FAIL] detected failed shard, cancelling all shard jobs"
    scancel "\${JOB_IDS[@]}" || true
    exit 1
  fi

  if [[ \$completed -eq \${#JOB_IDS[@]} ]]; then
    break
  fi

  sleep "\$WATCH_INTERVAL_SEC"
done

echo "[WATCHER] all shards completed, submitting CPU merge on \${CPU_LOGIN_HOST}"
merge_job_id="\$(ssh -o BatchMode=yes "\$CPU_LOGIN_HOST" "sbatch --parsable --account=\$ACCOUNT --partition=\$CPU_MERGE_PARTITION --cpus-per-task=\$CPU_MERGE_CPUS_PER_TASK --time=\$CPU_MERGE_TIME_LIMIT --export=ALL,PROJECT_ROOT=\$PROJECT_ROOT,OUTPUT_ROOT=\$OUTPUT_ROOT,FINAL_RUN_ID=\$FINAL_RUN_ID,BASE_RUN_ID=\$BASE_RUN_ID,SHARD_RUN_IDS=\$SHARD_RUN_IDS,SPLIT=\$SPLIT,QUESTION_LIMIT=\$QUESTION_LIMIT_FINAL,OVERWRITE_FINAL=1 \$MERGE_SCRIPT")"
echo "[WATCHER] cpu merge job submitted: \${merge_job_id}"
WATCHER_EOF

  chmod +x "${watcher_script}"
  watcher_pid="$(nohup bash "${watcher_script}" > "${watcher_log}" 2>&1 & echo $!)"

  echo "[INFO] postprocess mode=cpu_login"
  echo "[INFO] watcher_script=${watcher_script}"
  echo "[INFO] watcher_log=${watcher_log}"
  echo "[INFO] watcher_pid=${watcher_pid}"
  echo "[INFO] cpu_login_host=${CPU_LOGIN_HOST} cpu_partition=${CPU_MERGE_PARTITION}"

  echo "[INFO] queue snapshot (GPU shards only)"
  squeue -j "$(IFS=,; echo "${SHARD_JOB_IDS[*]}")" -o "%.18i %.10P %.28j %.10T %.12M %.10l %.6D %R"

  trap - EXIT
  exit 0
fi

# Legacy fallback: submit merge+guard on current scheduler (GPU side)
merge_job_id="$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name="stageD-d25-merge-${FINAL_RUN_ID}" \
  --partition="${MERGE_PARTITION}" \
  --cpus-per-task=8 \
  --gpus-per-node="${MERGE_GPUS_PER_NODE}" \
  --time="${MERGE_TIME_LIMIT}" \
  "${KILL_INVALID_DEP_FLAG[@]}" \
  --dependency="afterok:${afterok_ids}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},FINAL_RUN_ID=${FINAL_RUN_ID},BASE_RUN_ID=${BASE_RUN_ID},SHARD_RUN_IDS=${merge_shards},SPLIT=${SPLIT},QUESTION_LIMIT=${QUESTION_LIMIT_FINAL},OVERWRITE_FINAL=1" \
  "${MERGE_SCRIPT}")"

guard_dependency="afternotok:${SHARD_JOB_IDS[0]}?afternotok:${SHARD_JOB_IDS[1]}?afternotok:${SHARD_JOB_IDS[2]}?afternotok:${SHARD_JOB_IDS[3]}"
guard_job_id="$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name="stageD-d25-guard-${FINAL_RUN_ID}" \
  --partition="${GUARD_PARTITION}" \
  --cpus-per-task=1 \
  --gpus-per-node="${GUARD_GPUS_PER_NODE}" \
  --time="${GUARD_TIME_LIMIT}" \
  "${KILL_INVALID_DEP_FLAG[@]}" \
  --dependency="${guard_dependency}" \
  --output="/scratch/erichyu/kbp_test/%x-%j.out" \
  --error="/scratch/erichyu/kbp_test/%x-%j.err" \
  --wrap="echo '[GUARD] shard failed, cancelling remaining jobs'; if ! scancel ${SHARD_JOB_IDS[*]} ${merge_job_id}; then srun --jobid \$SLURM_JOB_ID --overlap bash -lc 'scancel ${SHARD_JOB_IDS[*]} ${merge_job_id}' || true; fi; date")"

echo "[INFO] postprocess mode=gpu_scheduler"
echo "[INFO] merge_job=${merge_job_id}"
echo "[INFO] guard_job=${guard_job_id}"
echo "[INFO] final_run_id=${FINAL_RUN_ID} base_run_id=${BASE_RUN_ID:-none} output_root=${OUTPUT_ROOT}"

echo "[INFO] queue snapshot"
squeue -j "$(IFS=,; echo "${SHARD_JOB_IDS[*]}")",${merge_job_id},${guard_job_id} -o "%.18i %.10P %.28j %.10T %.12M %.10l %.6D %R"

trap - EXIT
