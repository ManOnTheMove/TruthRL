#!/usr/bin/env python3
"""Local distillation via OpenAI-compatible endpoint (e.g., vLLM/SGLang)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CSV = REPO_ROOT / "data" / "medical" / "processed_csv" / "medqa_train_sft_half.csv"
DEFAULT_SAVE_DIR = REPO_ROOT / "data" / "medical" / "distill"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "medical" / "reports" / "stageC_distill_stats.json"
DEFAULT_FAILED_IDS = REPO_ROOT / "data" / "medical" / "reports" / "stageC_distill_failed_ids.txt"

SYSTEM_PROMPT = (
    "You are a medical expert with advanced clinical reasoning skills. "
    "Answer the multiple-choice question and follow this strict format:\n"
    "<think>...</think>\n"
    "<answer>...</answer>\n"
    "The final selected option MUST appear in \\boxed{} (for example, \\boxed{A})."
)


@dataclass
class DistillStats:
    total_rows: int = 0
    selected_rows: int = 0
    attempted_rows: int = 0
    success_rows: int = 0
    skipped_existing_rows: int = 0
    failed_rows: int = 0
    contract_fail_rows: int = 0
    api_fail_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local distillation for Stage C.")
    parser.add_argument("--sft_csv", type=Path, default=DEFAULT_SFT_CSV)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", type=str, default="Qwen3-0.6B")
    parser.add_argument("--api_key", type=str, default="token-abc123")
    parser.add_argument("--timeout_s", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--backoff_s", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--report_json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--failed_ids_path", type=Path, default=DEFAULT_FAILED_IDS)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force_target_boxed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail_on_error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    return args


def _parse_target_list(text: str) -> list[str]:
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("target must be a JSON list")
    return [str(x).strip() for x in value if str(x).strip()]


def load_sft_source_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SFT CSV: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question_id", "question_index", "problem", "target"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required fields: {required}")

        for row in reader:
            target = _parse_target_list(row["target"])
            rows.append(
                {
                    "question_id": str(row["question_id"]).strip(),
                    "question_index": int(row["question_index"]),
                    "problem": str(row["problem"]),
                    "target": target,
                }
            )
    return rows


def select_shard_rows(rows: list[dict[str, Any]], shard_index: int, num_shards: int, sample_limit: int | None) -> list[dict[str, Any]]:
    selected = [row for idx, row in enumerate(rows) if (idx % num_shards) == shard_index]
    if sample_limit is not None:
        selected = selected[:sample_limit]
    return selected


def build_teacher_messages(problem: str) -> list[dict[str, str]]:
    user_prompt = (
        "Question:\n"
        f"{problem.strip()}\n\n"
        "Please provide step-by-step clinical reasoning and then final answer. "
        "Final answer must include exactly one boxed option like \\boxed{A}."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_local_teacher_with_retry(
    endpoint: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str | None:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            content = parsed["choices"][0]["message"]["content"]
            if isinstance(content, str) and content.strip():
                return content.strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError):
            if attempt < max_retries:
                time.sleep(backoff_s * attempt)
            continue
    return None


def _extract_tag_content(text: str, tag: str) -> str | None:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_boxed_choice(text: str) -> str | None:
    match = re.search(r"\\boxed\s*\{\s*([^{}\s]+)\s*\}", text)
    if match:
        return match.group(1).strip().upper()
    return None


def normalize_completion_format(raw_text: str, target_choice: str | None, force_target_boxed: bool) -> str:
    raw = raw_text.strip()
    think_text = _extract_tag_content(raw, "think")
    answer_text = _extract_tag_content(raw, "answer")

    # Remove any existing structural tags so we can safely re-wrap without nested tags.
    plain_text = re.sub(r"</?(think|answer)>", "", raw, flags=re.IGNORECASE).strip()
    if think_text is None:
        think_text = plain_text or raw
    if answer_text is None:
        answer_text = plain_text or raw

    boxed = _extract_boxed_choice(raw)
    if boxed is None and force_target_boxed and target_choice:
        boxed = target_choice.strip().upper()

    if boxed and "\\boxed{" not in answer_text:
        answer_text = f"{answer_text.strip()}\nFinal answer: \\boxed{{{boxed}}}."

    normalized = f"<think>\n{think_text.strip()}\n</think>\n<answer>\n{answer_text.strip()}\n</answer>"
    return normalized


def validate_completion_contract(text: str) -> tuple[bool, str]:
    think_matches = re.findall(r"<think>.*?</think>", text, flags=re.DOTALL | re.IGNORECASE)
    answer_matches = re.findall(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    boxed_matches = re.findall(r"\\boxed\s*\{[^{}]+\}", text)

    if len(think_matches) != 1:
        return False, "think_tag_count_invalid"
    if len(answer_matches) != 1:
        return False, "answer_tag_count_invalid"
    if len(boxed_matches) < 1:
        return False, "boxed_missing"

    think_content = _extract_tag_content(text, "think")
    answer_content = _extract_tag_content(text, "answer")
    if not think_content:
        return False, "think_empty"
    if not answer_content:
        return False, "answer_empty"

    return True, "ok"


def write_completion_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_distill_summary(summary: dict[str, Any], report_path: Path, failed_ids_path: Path, failed_ids: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    failed_ids_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with failed_ids_path.open("w", encoding="utf-8") as f:
        for qid in failed_ids:
            f.write(f"{qid}\n")


def main() -> None:
    args = parse_args()
    rows = load_sft_source_csv(args.sft_csv)
    shard_rows = select_shard_rows(rows, args.shard_index, args.num_shards, args.sample_limit)

    stats = DistillStats(total_rows=len(rows), selected_rows=len(shard_rows))
    failed_ids: list[str] = []

    for row in shard_rows:
        question_id = row["question_id"]
        out_path = args.save_dir / f"{question_id}.txt"

        if out_path.exists() and not args.overwrite:
            stats.skipped_existing_rows += 1
            continue

        stats.attempted_rows += 1
        target_choice = row["target"][0] if row["target"] else None
        messages = build_teacher_messages(row["problem"])

        raw_content = call_local_teacher_with_retry(
            endpoint=args.endpoint,
            model=args.model,
            api_key=args.api_key,
            messages=messages,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            backoff_s=args.backoff_s,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        if raw_content is None:
            stats.failed_rows += 1
            stats.api_fail_rows += 1
            failed_ids.append(question_id)
            continue

        normalized = normalize_completion_format(raw_content, target_choice, args.force_target_boxed)
        valid, reason = validate_completion_contract(normalized)
        if not valid:
            stats.failed_rows += 1
            stats.contract_fail_rows += 1
            failed_ids.append(question_id)
            continue

        write_completion_atomic(out_path, normalized + "\n")
        stats.success_rows += 1

    summary = {
        "sft_csv": str(args.sft_csv),
        "save_dir": str(args.save_dir),
        "endpoint": args.endpoint,
        "model": args.model,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sample_limit": args.sample_limit,
        "stats": {
            "total_rows": stats.total_rows,
            "selected_rows": stats.selected_rows,
            "attempted_rows": stats.attempted_rows,
            "success_rows": stats.success_rows,
            "skipped_existing_rows": stats.skipped_existing_rows,
            "failed_rows": stats.failed_rows,
            "api_fail_rows": stats.api_fail_rows,
            "contract_fail_rows": stats.contract_fail_rows,
        },
        "failed_ids_count": len(failed_ids),
        "failed_ids_path": str(args.failed_ids_path),
    }

    write_distill_summary(summary, args.report_json, args.failed_ids_path, failed_ids)

    print("[INFO] Local distill finished.")
    print(f"  selected_rows={stats.selected_rows}")
    print(f"  attempted_rows={stats.attempted_rows}")
    print(f"  success_rows={stats.success_rows}")
    print(f"  failed_rows={stats.failed_rows}")
    print(f"  skipped_existing_rows={stats.skipped_existing_rows}")
    print(f"  report_json={args.report_json}")

    if args.fail_on_error and stats.failed_rows > 0:
        raise SystemExit(f"Distillation failed for {stats.failed_rows} rows. See {args.failed_ids_path}")


if __name__ == "__main__":
    main()
