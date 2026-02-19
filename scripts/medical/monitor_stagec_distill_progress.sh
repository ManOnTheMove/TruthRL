#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SFT_CSV="${SFT_CSV:-${REPO_ROOT}/data/medical/processed_csv/medqa_train_sft_half.csv}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/data/medical/distill}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/data/medical/reports}"
JOB_ID="${1:-${JOB_ID:-}}"
WATCH_MODE="${WATCH_MODE:-0}"
INTERVAL_S="${INTERVAL_S:-30}"

if [[ ! -f "${SFT_CSV}" ]]; then
  echo "[ERROR] Missing SFT_CSV: ${SFT_CSV}"
  exit 1
fi

if [[ -n "${JOB_ID}" ]]; then
  PASS_REPORT_DIR="${PASS_REPORT_DIR:-${REPORT_DIR}/stageC_distill_passes_${JOB_ID}}"
else
  PASS_REPORT_DIR="${PASS_REPORT_DIR:-${REPORT_DIR}}"
fi

print_once() {
  python3 - <<PY
import csv
import glob
import json
from pathlib import Path
from datetime import datetime

sft_csv = Path(r"${SFT_CSV}")
save_dir = Path(r"${SAVE_DIR}")
pass_report_dir = Path(r"${PASS_REPORT_DIR}")
job_id = r"${JOB_ID}"

total = 0
with sft_csv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for _ in reader:
        total += 1

done_ids = set()
if save_dir.exists():
    for p in save_dir.glob("*.txt"):
        done_ids.add(p.stem)

done = len(done_ids)
missing = max(total - done, 0)
pct = (done / total * 100.0) if total else 0.0

print(f"[{datetime.now().isoformat(timespec='seconds')}]")
print(f"total_rows={total}")
print(f"done_txt={done}")
print(f"missing_estimate={missing}")
print(f"progress={pct:.2f}%")
print(f"distill_dir={save_dir}")

if job_id:
    print(f"job_id={job_id}")
    print(f"pass_report_dir={pass_report_dir}")

summaries = sorted(glob.glob(str(pass_report_dir / "pass_*_summary.json")))
if summaries:
    latest = Path(summaries[-1])
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        print(f"latest_pass_summary={latest}")
        print(f"latest_pass_id={data.get('pass_id')}")
        print(f"latest_missing_after_pass={data.get('missing_count_after_pass')}")
    except Exception:
        print(f"latest_pass_summary={latest} (unreadable)")
else:
    print("latest_pass_summary=not_found_yet")

failed_ids = Path(r"${REPORT_DIR}") / "stageC_distill_failed_ids.txt"
if failed_ids.exists():
    try:
        n_failed = sum(1 for _ in failed_ids.open("r", encoding="utf-8"))
        print(f"final_failed_ids_count={n_failed}")
        print(f"final_failed_ids_path={failed_ids}")
    except Exception:
        print(f"final_failed_ids_path={failed_ids} (unreadable)")
else:
    print("final_failed_ids_path=not_generated_yet")
PY
}

if [[ "${WATCH_MODE}" == "1" ]]; then
  while true; do
    print_once
    echo "---"
    sleep "${INTERVAL_S}"
  done
else
  print_once
fi

