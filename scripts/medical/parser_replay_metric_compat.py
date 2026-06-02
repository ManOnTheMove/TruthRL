#!/usr/bin/env python3
"""Replay medical parser compatibility on historical KBP and Stage-D outputs.

This is an offline analysis tool. It reads existing response/prediction files,
recomputes parser outcomes with the current shared parser, and writes small
CSV/JSON/Markdown summaries. It does not generate model outputs or modify
historical KBP/eval artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_KBP_RUN_ID = "kbp_20260319_stagec-c8-8b_medqa_grpo_train_fullpass"
DEFAULT_STAGE_D_EVAL_ROOT = Path(
    "/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/"
    "phase23_20260511_standard_d26_stagec_vs_d3"
)

POLICY_KBP_NEW = "final_answer_last_boxed"
POLICY_STAGE_D_NEW = "single_boxed_strict"
POLICY_LEGACY_FIRST = "first_boxed_legacy"

OLD_BOXED_PATTERN = re.compile(r"\\boxed\s*{(.*?)}", re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
OLD_OPTION_LABEL_PATTERN = re.compile(r"^\s*([A-Za-z])\s*[:.)\]-]?\s*")
OPTION_TEXT_PATTERN = re.compile(r"^\s*[A-Za-z]\s*(?:[).:]|-\s+).+")

BOXED_VALID_STATUSES = {"ok_choice", "ok_abstain"}
BUCKET_ORDER = ["0", "1-63", "64-127", "128-191", "192-255", "256", "other"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Replay Phase-C parser metric compatibility.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--kbp-run-dir", type=Path, default=None)
    parser.add_argument("--kbp-split", type=str, default="")
    parser.add_argument("--input-parquet", type=Path, default=None)
    parser.add_argument("--stageD-eval-root", type=Path, default=DEFAULT_STAGE_D_EVAL_ROOT)
    parser.add_argument("--stageD-model-ids", type=str, default="stagec_base,d3_global_step_300")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-kbp-responses", type=int, default=0)
    parser.add_argument("--max-stageD-rows", type=int, default=0)
    parser.add_argument("--example-limit", type=int, default=200)
    parser.add_argument("--skip-kbp", action="store_true")
    parser.add_argument("--skip-stageD", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def path_from_string(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def resolve_existing_path(path: Path, repo_root: Path) -> Path:
    if path.exists():
        return path

    text = str(path)
    project_root = repo_root.parent
    replacements = [
        ("/home/erichyu/links/projects/def-zshakeri/erichyu/embc", str(project_root)),
        ("/home/erichyu/projects/def-zshakeri/erichyu/embc", str(project_root)),
        ("/project/def-zshakeri/erichyu/embc", str(project_root)),
    ]
    for old, new in replacements:
        if text.startswith(old):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
    return path


def default_out_dir(repo_root: Path, is_smoke: bool) -> Path:
    suffix = "smoke" if is_smoke else utc_stamp()
    return repo_root / "data" / "medical" / "parser_replay" / f"phaseC_parser_compat_{suffix}"


def load_parser_module(repo_root: Path) -> Any:
    parser_path = repo_root / "training" / "verl" / "verl" / "utils" / "reward_score" / "medical_answer_parser.py"
    if not parser_path.exists():
        raise FileNotFoundError(f"medical answer parser not found: {parser_path}")
    spec = importlib.util.spec_from_file_location("medical_answer_parser_replay", str(parser_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load parser module from: {parser_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_head(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def old_normalize_choice(text: str) -> str:
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    match = OLD_OPTION_LABEL_PATTERN.match(stripped)
    if match:
        return match.group(1).upper()
    compact = re.sub(r"\s+", "", stripped).upper()
    if len(compact) == 1 and "A" <= compact <= "Z":
        return compact
    return ""


def old_is_abstain_boxed(text: str) -> bool:
    if not isinstance(text, str):
        return False
    canonical = re.sub(r"[^a-z0-9]+", "", text.lower())
    return canonical in {"idontknow", "idonotknow", "idk"}


def normalize_target_set(parser_module: Any, ground_truth: dict[str, Any]) -> set[str]:
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    out: set[str] = set()
    if isinstance(targets, list):
        for item in targets:
            value = parser_module.normalize_choice(str(item))
            if value:
                out.add(value)
    return out


def normalize_choice_set(parser_module: Any, ground_truth: dict[str, Any]) -> set[str]:
    choices = ground_truth.get("choices", [])
    out: set[str] = set()
    if isinstance(choices, list):
        for item in choices:
            if not isinstance(item, str):
                continue
            value = parser_module.normalize_choice(item)
            if value:
                out.add(value)
                continue
            if ":" in item:
                value = parser_module.normalize_choice(item.split(":", 1)[0])
                if value:
                    out.add(value)
    return out


def load_target_choice_maps(
    input_parquet: Path,
    parser_module: Any,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    table = pq.read_table(input_parquet, columns=["reward_model", "extra_info"])
    target_map: dict[str, set[str]] = {}
    choice_map: dict[str, set[str]] = {}
    for idx, row in enumerate(table.to_pylist()):
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") or {}
        extra_info = row.get("extra_info") or {}
        qid = str(extra_info.get("question_id", f"row_{idx}"))
        target_map[qid] = normalize_target_set(parser_module, ground_truth)
        choice_map[qid] = normalize_choice_set(parser_module, ground_truth)
    return target_map, choice_map


def difficulty_bucket(correct_count: int) -> str:
    count = int(correct_count)
    if count == 0:
        return "0"
    if 1 <= count <= 63:
        return "1-63"
    if 64 <= count <= 127:
        return "64-127"
    if 128 <= count <= 191:
        return "128-191"
    if 192 <= count <= 255:
        return "192-255"
    if count == 256:
        return "256"
    return "other"


def parse_fail_status(status: str) -> bool:
    return str(status) not in BOXED_VALID_STATUSES


def parse_valid_status(status: str) -> bool:
    return str(status) in BOXED_VALID_STATUSES


def old_extract_kbp_final_boxed(response_text: str) -> tuple[bool, str, int, bool]:
    text = response_text if isinstance(response_text, str) else ""
    global_matches = [x.strip() for x in OLD_BOXED_PATTERN.findall(text)]
    answer_blocks = ANSWER_TAG_PATTERN.findall(text)
    if answer_blocks:
        answer_matches = [x.strip() for x in OLD_BOXED_PATTERN.findall(answer_blocks[-1])]
        if answer_matches:
            return True, answer_matches[-1], len(global_matches), True
    if global_matches:
        return True, global_matches[-1], len(global_matches), False
    return False, "", 0, False


def old_parse_selected(
    boxed_raw: str,
    *,
    has_boxed: bool,
    boxed_count: int,
    used_answer_block: bool,
    target_set: set[str],
    choice_set: set[str],
    policy: str,
) -> dict[str, Any]:
    if not has_boxed:
        return {
            "boxed_raw": "",
            "normalized_choice": "",
            "is_abstain": 0,
            "is_correct": 0,
            "parse_status": "no_boxed",
            "parser_policy": policy,
            "boxed_count": int(boxed_count),
            "used_answer_block": int(used_answer_block),
            "is_boxed_valid": 0,
        }

    if old_is_abstain_boxed(boxed_raw):
        return {
            "boxed_raw": boxed_raw,
            "normalized_choice": "I don't know",
            "is_abstain": 1,
            "is_correct": 0,
            "parse_status": "ok_abstain",
            "parser_policy": policy,
            "boxed_count": int(boxed_count),
            "used_answer_block": int(used_answer_block),
            "is_boxed_valid": 1,
        }

    normalized = old_normalize_choice(boxed_raw)
    if not normalized:
        return {
            "boxed_raw": boxed_raw,
            "normalized_choice": "",
            "is_abstain": 0,
            "is_correct": 0,
            "parse_status": "invalid_choice",
            "parser_policy": policy,
            "boxed_count": int(boxed_count),
            "used_answer_block": int(used_answer_block),
            "is_boxed_valid": 0,
        }

    if choice_set and normalized not in choice_set:
        return {
            "boxed_raw": boxed_raw,
            "normalized_choice": normalized,
            "is_abstain": 0,
            "is_correct": 0,
            "parse_status": "choice_out_of_set",
            "parser_policy": policy,
            "boxed_count": int(boxed_count),
            "used_answer_block": int(used_answer_block),
            "is_boxed_valid": 0,
        }

    return {
        "boxed_raw": boxed_raw,
        "normalized_choice": normalized,
        "is_abstain": 0,
        "is_correct": int(normalized in target_set),
        "parse_status": "ok_choice",
        "parser_policy": policy,
        "boxed_count": int(boxed_count),
        "used_answer_block": int(used_answer_block),
        "is_boxed_valid": 1,
    }


def old_parse_kbp(response_text: str, target_set: set[str], choice_set: set[str]) -> dict[str, Any]:
    has_boxed, boxed_raw, boxed_count, used_answer_block = old_extract_kbp_final_boxed(response_text)
    return old_parse_selected(
        boxed_raw,
        has_boxed=has_boxed,
        boxed_count=boxed_count,
        used_answer_block=used_answer_block,
        target_set=target_set,
        choice_set=choice_set,
        policy="old_regex_final_answer_last_boxed",
    )


def old_parse_strict(response_text: str, target_set: set[str], choice_set: set[str]) -> dict[str, Any]:
    text = response_text if isinstance(response_text, str) else ""
    matches = [x.strip() for x in OLD_BOXED_PATTERN.findall(text)]
    if len(matches) == 0:
        return old_parse_selected(
            "",
            has_boxed=False,
            boxed_count=0,
            used_answer_block=False,
            target_set=target_set,
            choice_set=choice_set,
            policy="old_regex_single_boxed_strict",
        )
    if len(matches) > 1:
        return {
            "boxed_raw": " | ".join(matches),
            "normalized_choice": "",
            "is_abstain": 0,
            "is_correct": 0,
            "parse_status": "multi_boxed",
            "parser_policy": "old_regex_single_boxed_strict",
            "boxed_count": int(len(matches)),
            "used_answer_block": 0,
            "is_boxed_valid": 0,
        }
    return old_parse_selected(
        matches[0],
        has_boxed=True,
        boxed_count=1,
        used_answer_block=False,
        target_set=target_set,
        choice_set=choice_set,
        policy="old_regex_single_boxed_strict",
    )


def old_parse_first(response_text: str, target_set: set[str], choice_set: set[str]) -> dict[str, Any]:
    text = response_text if isinstance(response_text, str) else ""
    matches = [x.strip() for x in OLD_BOXED_PATTERN.findall(text)]
    if not matches:
        return old_parse_selected(
            "",
            has_boxed=False,
            boxed_count=0,
            used_answer_block=False,
            target_set=target_set,
            choice_set=choice_set,
            policy="old_regex_first_boxed_legacy",
        )
    return old_parse_selected(
        matches[0],
        has_boxed=True,
        boxed_count=len(matches),
        used_answer_block=False,
        target_set=target_set,
        choice_set=choice_set,
        policy="old_regex_first_boxed_legacy",
    )


def new_parse(
    parser_module: Any,
    response_text: str,
    *,
    policy: str,
    target_set: set[str],
    choice_set: set[str],
) -> dict[str, Any]:
    parsed = parser_module.parse_medical_answer(response_text or "", policy=policy, choice_set=choice_set)
    is_correct = int(parsed.parse_status == "ok_choice" and parsed.normalized_choice in target_set)
    return {
        "boxed_raw": parsed.boxed_raw,
        "normalized_choice": parsed.normalized_choice,
        "is_abstain": int(parsed.is_abstain),
        "is_correct": is_correct,
        "parse_status": parsed.parse_status,
        "parser_policy": parsed.policy,
        "boxed_count": int(parsed.boxed_count),
        "used_answer_block": int(parsed.used_answer_block),
        "is_boxed_valid": int(parsed.parse_status in BOXED_VALID_STATUSES),
    }


def historical_kbp_parse(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("parse_status", ""))
    return {
        "boxed_raw": str(row.get("boxed_raw", "") or ""),
        "normalized_choice": str(row.get("normalized_choice", "") or ""),
        "is_abstain": int(row.get("is_abstain", 0) or 0),
        "is_correct": int(row.get("is_correct", 0) or 0),
        "parse_status": status,
        "parser_policy": "historical_persisted",
        "boxed_count": "",
        "used_answer_block": "",
        "is_boxed_valid": int(status in BOXED_VALID_STATUSES),
    }


def historical_stage_parse(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "boxed_raw": str(row.get("boxed_raw", "") or ""),
        "normalized_choice": str(row.get("normalized_choice", "") or ""),
        "is_abstain": int(row.get("is_abstain", 0) or 0),
        "is_correct": int(row.get("is_correct", 0) or 0),
        "parse_status": str(row.get("parse_status", "") or ""),
        "parser_policy": "historical_persisted",
        "boxed_count": "",
        "used_answer_block": "",
        "is_boxed_valid": int(row.get("is_boxed_valid", 0) or 0),
    }


def flip_category(old: dict[str, Any], new: dict[str, Any]) -> str:
    old_correct = bool(int(old.get("is_correct", 0) or 0))
    new_correct = bool(int(new.get("is_correct", 0) or 0))
    old_abstain = bool(int(old.get("is_abstain", 0) or 0))
    new_abstain = bool(int(new.get("is_abstain", 0) or 0))
    old_status = str(old.get("parse_status", ""))
    new_status = str(new.get("parse_status", ""))

    if (
        old_correct == new_correct
        and old_abstain == new_abstain
        and old_status == new_status
        and str(old.get("normalized_choice", "")) == str(new.get("normalized_choice", ""))
    ):
        return "same"

    if old_correct and not new_correct:
        if parse_fail_status(new_status):
            return "old_correct_new_parse_fail"
        if new_abstain:
            return "old_correct_new_abstain"
        return "old_correct_new_wrong"

    if not old_correct and new_correct:
        if parse_fail_status(old_status):
            return "old_parse_fail_new_correct"
        return "old_wrong_new_correct"

    if old_abstain != new_abstain:
        return "abstain_status_changed"

    if parse_fail_status(old_status) != parse_fail_status(new_status):
        return "parse_fail_status_changed"

    if old_status != new_status:
        return "parse_status_changed_only"

    if str(old.get("normalized_choice", "")) != str(new.get("normalized_choice", "")):
        return "normalized_choice_changed_only"

    return "same_outcome"


def special_flags(parser_module: Any, response_text: str, old_row: dict[str, Any], new_row: dict[str, Any]) -> dict[str, int]:
    text = response_text if isinstance(response_text, str) else ""
    old_raw = str(old_row.get("boxed_raw", "") or "")
    new_raw = str(new_row.get("boxed_raw", "") or "")
    raw = new_raw or old_raw
    old_norm = old_normalize_choice(raw)
    new_norm = parser_module.normalize_choice(raw)
    return {
        "contains_aspirin_like": int(bool(re.search(r"\\boxed\s*{\s*Aspirin\b", text, re.IGNORECASE))),
        "contains_option_text_boxed": int(bool(OPTION_TEXT_PATTERN.match(raw))),
        "old_text_normalization_risk": int(bool(old_norm and not new_norm and re.match(r"^[A-Za-z]{2,}", raw.strip()))),
        "multi_boxed": int(int(new_row.get("boxed_count", 0) or 0) > 1 or len(OLD_BOXED_PATTERN.findall(text)) > 1),
        "nested_braces": int(bool(re.search(r"\\boxed\s*{[^}]*{", text))),
        "used_answer_block": int(int(new_row.get("used_answer_block", 0) or 0)),
    }


def qstats() -> dict[str, Any]:
    return {
        "probe_count": 0,
        "historical_correct_count": 0,
        "historical_abstain_count": 0,
        "historical_parse_fail_count": 0,
        "old_compat_correct_count": 0,
        "old_compat_abstain_count": 0,
        "old_compat_parse_fail_count": 0,
        "new_correct_count": 0,
        "new_abstain_count": 0,
        "new_parse_fail_count": 0,
        "response_outcome_changed_count": 0,
        "parse_status_changed_count": 0,
        "normalized_choice_changed_count": 0,
    }


def update_question_stats(
    stats: dict[str, Any],
    historical: dict[str, Any],
    old_compat: dict[str, Any],
    new: dict[str, Any],
    category: str,
) -> None:
    stats["probe_count"] += 1
    stats["historical_correct_count"] += int(historical.get("is_correct", 0) or 0)
    stats["historical_abstain_count"] += int(historical.get("is_abstain", 0) or 0)
    stats["historical_parse_fail_count"] += int(parse_fail_status(str(historical.get("parse_status", ""))))
    stats["old_compat_correct_count"] += int(old_compat.get("is_correct", 0) or 0)
    stats["old_compat_abstain_count"] += int(old_compat.get("is_abstain", 0) or 0)
    stats["old_compat_parse_fail_count"] += int(parse_fail_status(str(old_compat.get("parse_status", ""))))
    stats["new_correct_count"] += int(new.get("is_correct", 0) or 0)
    stats["new_abstain_count"] += int(new.get("is_abstain", 0) or 0)
    stats["new_parse_fail_count"] += int(parse_fail_status(str(new.get("parse_status", ""))))
    stats["response_outcome_changed_count"] += int(category != "same")
    stats["parse_status_changed_count"] += int(str(historical.get("parse_status", "")) != str(new.get("parse_status", "")))
    stats["normalized_choice_changed_count"] += int(
        str(historical.get("normalized_choice", "")) != str(new.get("normalized_choice", ""))
    )


def summarize_parse_source(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    n = len(materialized)
    correct = sum(int(row.get("is_correct", 0) or 0) for row in materialized)
    abstain = sum(int(row.get("is_abstain", 0) or 0) for row in materialized)
    boxed_valid = sum(int(row.get("is_boxed_valid", 0) or 0) for row in materialized)
    valid_answers = sum(1 for row in materialized if str(row.get("parse_status", "")) == "ok_choice")
    parse_fail = n - boxed_valid
    valid_wrong = sum(
        1
        for row in materialized
        if str(row.get("parse_status", "")) == "ok_choice" and not int(row.get("is_correct", 0) or 0)
    )
    return {
        "n": n,
        "correct_count": correct,
        "accuracy": correct / n if n else 0.0,
        "valid_answer_count": valid_answers,
        "coverage": valid_answers / n if n else 0.0,
        "selective_accuracy": correct / valid_answers if valid_answers else "",
        "abstain_count": abstain,
        "abstain_rate": abstain / n if n else 0.0,
        "boxed_valid_count": boxed_valid,
        "boxed_valid_rate": boxed_valid / n if n else 0.0,
        "parse_fail_count": parse_fail,
        "parse_fail_rate": parse_fail / n if n else 0.0,
        "valid_non_abstain_wrong_count": valid_wrong,
        "answered_error_rate": valid_wrong / valid_answers if valid_answers else "",
        "parse_status_counts": dict(Counter(str(row.get("parse_status", "")) for row in materialized)),
    }


def resolve_kbp_inputs(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    run_dir = args.kbp_run_dir or (repo_root / "data" / "medical" / "kbp" / "runs" / DEFAULT_KBP_RUN_ID)
    run_dir = resolve_existing_path(run_dir, repo_root)
    summary_path = run_dir / "reports" / "kbp_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}

    split = args.kbp_split or str(summary.get("split") or "medqa_grpo_train")
    input_parquet = args.input_parquet or path_from_string(summary.get("input_parquet"))
    if input_parquet is None:
        input_parquet = repo_root / "data" / "medical" / "verl" / "medqa_grpo_train.parquet"
    input_parquet = resolve_existing_path(input_parquet, repo_root)

    responses_dir = run_dir / "responses" / f"split={split}"
    parsed_dir = run_dir / "parsed" / f"split={split}"
    response_parts = sorted(responses_dir.glob("part-*.parquet"))
    parsed_parts = sorted(parsed_dir.glob("part-*.parquet"))
    labels_path = run_dir / "labels" / "kbp_labels_by_question.parquet"

    if not response_parts:
        raise FileNotFoundError(f"no KBP response parts found under: {responses_dir}")
    if not parsed_parts:
        raise FileNotFoundError(f"no KBP parsed parts found under: {parsed_dir}")
    if len(response_parts) != len(parsed_parts):
        raise RuntimeError(f"KBP response/parsed part count mismatch: {len(response_parts)} vs {len(parsed_parts)}")

    return {
        "run_dir": run_dir,
        "summary_path": summary_path,
        "summary": summary,
        "split": split,
        "input_parquet": input_parquet,
        "response_parts": response_parts,
        "parsed_parts": parsed_parts,
        "labels_path": labels_path,
    }


def replay_kbp(
    *,
    args: argparse.Namespace,
    parser_module: Any,
    out_dir: Path,
) -> dict[str, Any]:
    inputs = resolve_kbp_inputs(args)
    target_map, choice_map = load_target_choice_maps(inputs["input_parquet"], parser_module)

    parse_counts = {
        "historical_persisted": Counter(),
        "old_compat_final_answer_last_boxed": Counter(),
        "new_final_answer_last_boxed": Counter(),
    }
    flip_counts: Counter[str] = Counter()
    special_counts: Counter[str] = Counter()
    qid_stats: dict[str, dict[str, Any]] = defaultdict(qstats)
    processed = 0
    part_mismatch_count = 0

    examples_path = out_dir / "kbp_response_flip_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as examples_f:
        for response_path, parsed_path in zip(inputs["response_parts"], inputs["parsed_parts"]):
            response_table = pq.ParquetFile(response_path).read(
                columns=["question_id", "probe_index", "response_text"]
            )
            parsed_table = pq.ParquetFile(parsed_path).read(
                columns=["question_id", "probe_index", "boxed_raw", "normalized_choice", "is_abstain", "is_correct", "parse_status"]
            )
            if response_table.num_rows != parsed_table.num_rows:
                raise RuntimeError(f"row mismatch: {response_path} vs {parsed_path}")

            response_rows = response_table.to_pylist()
            parsed_rows = parsed_table.to_pylist()
            for response_row, parsed_row in zip(response_rows, parsed_rows):
                if args.max_kbp_responses > 0 and processed >= args.max_kbp_responses:
                    break

                qid = str(response_row.get("question_id") or "")
                probe_index = int(response_row.get("probe_index", -1) or -1)
                if qid != str(parsed_row.get("question_id") or "") or probe_index != int(parsed_row.get("probe_index", -1) or -1):
                    part_mismatch_count += 1

                response_text = response_row.get("response_text")
                response_text = response_text if isinstance(response_text, str) else ""
                targets = target_map.get(qid, set())
                choices = choice_map.get(qid, set())

                historical = historical_kbp_parse(parsed_row)
                old_compat = old_parse_kbp(response_text, targets, choices)
                new = new_parse(
                    parser_module,
                    response_text,
                    policy=POLICY_KBP_NEW,
                    target_set=targets,
                    choice_set=choices,
                )

                parse_counts["historical_persisted"][str(historical["parse_status"])] += 1
                parse_counts["old_compat_final_answer_last_boxed"][str(old_compat["parse_status"])] += 1
                parse_counts["new_final_answer_last_boxed"][str(new["parse_status"])] += 1

                category = flip_category(historical, new)
                flip_counts[category] += 1
                flags = special_flags(parser_module, response_text, historical, new)
                for key, value in flags.items():
                    special_counts[key] += int(value)

                update_question_stats(qid_stats[qid], historical, old_compat, new, category)

                if category != "same" and sum(v for k, v in flip_counts.items() if k != "same") <= args.example_limit:
                    append_jsonl(
                        examples_f,
                        {
                            "question_id": qid,
                            "probe_index": probe_index,
                            "flip_category": category,
                            "historical": historical,
                            "old_compat": old_compat,
                            "new": new,
                            "special_flags": flags,
                            "response_head": response_text[:900].replace("\n", "\\n"),
                            "response_tail": response_text[-500:].replace("\n", "\\n"),
                        },
                    )
                processed += 1

            if args.max_kbp_responses > 0 and processed >= args.max_kbp_responses:
                break

    question_rows: list[dict[str, Any]] = []
    question_change_counts: Counter[str] = Counter()
    for qid in sorted(qid_stats):
        stats = qid_stats[qid]
        old_correct = int(stats["historical_correct_count"])
        new_correct = int(stats["new_correct_count"])
        old_ook = old_correct == 0
        new_ook = new_correct == 0
        old_bucket = difficulty_bucket(old_correct)
        new_bucket = difficulty_bucket(new_correct)
        row = {
            "question_id": qid,
            "probe_count": int(stats["probe_count"]),
            "historical_correct_count": old_correct,
            "old_compat_correct_count": int(stats["old_compat_correct_count"]),
            "new_correct_count": new_correct,
            "correct_count_delta_new_minus_historical": new_correct - old_correct,
            "historical_abstain_count": int(stats["historical_abstain_count"]),
            "old_compat_abstain_count": int(stats["old_compat_abstain_count"]),
            "new_abstain_count": int(stats["new_abstain_count"]),
            "historical_parse_fail_count": int(stats["historical_parse_fail_count"]),
            "old_compat_parse_fail_count": int(stats["old_compat_parse_fail_count"]),
            "new_parse_fail_count": int(stats["new_parse_fail_count"]),
            "historical_ook": int(old_ook),
            "new_ook": int(new_ook),
            "ook_label_changed": int(old_ook != new_ook),
            "historical_difficulty_bucket": old_bucket,
            "new_difficulty_bucket": new_bucket,
            "difficulty_bucket_changed": int(old_bucket != new_bucket),
            "response_outcome_changed_count": int(stats["response_outcome_changed_count"]),
            "parse_status_changed_count": int(stats["parse_status_changed_count"]),
            "normalized_choice_changed_count": int(stats["normalized_choice_changed_count"]),
            "abstain_count_changed": int(stats["historical_abstain_count"] != stats["new_abstain_count"]),
        }
        question_rows.append(row)
        question_change_counts["correct_count_changed"] += int(old_correct != new_correct)
        question_change_counts["ook_label_changed"] += int(row["ook_label_changed"])
        question_change_counts["difficulty_bucket_changed"] += int(row["difficulty_bucket_changed"])
        question_change_counts["abstain_count_changed"] += int(row["abstain_count_changed"])

    labels_match = ""
    if args.max_kbp_responses == 0 and inputs["labels_path"].exists():
        label_rows = pq.read_table(inputs["labels_path"]).to_pylist()
        q_by_id = {row["question_id"]: row for row in question_rows}
        labels_match = all(
            str(label.get("question_id")) in q_by_id
            and int(label.get("correct_count", -1)) == int(q_by_id[str(label.get("question_id"))]["historical_correct_count"])
            and int(label.get("abstain_count", -1)) == int(q_by_id[str(label.get("question_id"))]["historical_abstain_count"])
            for label in label_rows
        )

    question_path = out_dir / "kbp_question_flips.csv"
    write_csv(
        question_path,
        question_rows,
        [
            "question_id",
            "probe_count",
            "historical_correct_count",
            "old_compat_correct_count",
            "new_correct_count",
            "correct_count_delta_new_minus_historical",
            "historical_abstain_count",
            "old_compat_abstain_count",
            "new_abstain_count",
            "historical_parse_fail_count",
            "old_compat_parse_fail_count",
            "new_parse_fail_count",
            "historical_ook",
            "new_ook",
            "ook_label_changed",
            "historical_difficulty_bucket",
            "new_difficulty_bucket",
            "difficulty_bucket_changed",
            "response_outcome_changed_count",
            "parse_status_changed_count",
            "normalized_choice_changed_count",
            "abstain_count_changed",
        ],
    )

    return {
        "kind": "kbp",
        "run_dir": str(inputs["run_dir"]),
        "split": inputs["split"],
        "input_parquet": str(inputs["input_parquet"]),
        "processed_responses": int(processed),
        "max_kbp_responses": int(args.max_kbp_responses),
        "part_mismatch_count": int(part_mismatch_count),
        "parse_status_counts": {key: dict(value) for key, value in parse_counts.items()},
        "flip_counts": dict(flip_counts),
        "special_counts": dict(special_counts),
        "question_count": int(len(question_rows)),
        "question_change_counts": dict(question_change_counts),
        "labels_match_historical_counts": labels_match,
        "outputs": {
            "kbp_question_flips_csv": str(question_path),
            "kbp_response_flip_examples_jsonl": str(examples_path),
        },
    }


def grouped_stage_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["all"] = list(rows)
    for row in rows:
        grouped[str(row.get("source_group", ""))].append(row)
    return grouped


def replay_stageD(
    *,
    args: argparse.Namespace,
    parser_module: Any,
    out_dir: Path,
) -> dict[str, Any]:
    model_ids = [item.strip() for item in args.stageD_model_ids.split(",") if item.strip()]
    metric_rows: list[dict[str, Any]] = []
    parse_compare_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    examples_path = out_dir / "stageD_response_flip_examples.jsonl"
    examples_written = 0

    with examples_path.open("w", encoding="utf-8") as examples_f:
        for model_id in model_ids:
            pred_path = args.stageD_eval_root / model_id / "predictions.parquet"
            if not pred_path.exists():
                raise FileNotFoundError(f"StageD predictions not found for model_id={model_id}: {pred_path}")
            table = pq.read_table(
                pred_path,
                columns=[
                    "source_group",
                    "question_id",
                    "target_choices",
                    "choice_set",
                    "response_text",
                    "boxed_raw",
                    "normalized_choice",
                    "is_abstain",
                    "is_boxed_valid",
                    "is_correct",
                    "parse_status",
                ],
            )
            rows = table.to_pylist()
            if args.max_stageD_rows > 0:
                rows = rows[: args.max_stageD_rows]

            historical_rows: list[dict[str, Any]] = []
            old_compat_rows: list[dict[str, Any]] = []
            new_rows: list[dict[str, Any]] = []
            flip_counts: Counter[str] = Counter()

            for row in rows:
                targets = {str(x) for x in (row.get("target_choices") or [])}
                choices = {str(x) for x in (row.get("choice_set") or [])}
                response_text = row.get("response_text")
                response_text = response_text if isinstance(response_text, str) else ""
                historical = historical_stage_parse(row)
                old_compat = old_parse_strict(response_text, targets, choices)
                new = new_parse(
                    parser_module,
                    response_text,
                    policy=POLICY_STAGE_D_NEW,
                    target_set=targets,
                    choice_set=choices,
                )
                for parsed in [historical, old_compat, new]:
                    parsed["source_group"] = str(row.get("source_group", ""))

                category = flip_category(historical, new)
                flip_counts[category] += 1
                historical_rows.append(historical)
                old_compat_rows.append(old_compat)
                new_rows.append(new)

                if category != "same" and examples_written < args.example_limit:
                    append_jsonl(
                        examples_f,
                        {
                            "model_id": model_id,
                            "source_group": str(row.get("source_group", "")),
                            "question_id": str(row.get("question_id", "")),
                            "flip_category": category,
                            "historical": historical,
                            "old_compat": old_compat,
                            "new": new,
                            "response_head": response_text[:900].replace("\n", "\\n"),
                            "response_tail": response_text[-500:].replace("\n", "\\n"),
                        },
                    )
                    examples_written += 1

            source_sets = {
                "historical_persisted": historical_rows,
                "old_compat_single_boxed_strict": old_compat_rows,
                "new_single_boxed_strict": new_rows,
            }
            for source_name, source_rows in source_sets.items():
                for group_name, group_rows in grouped_stage_rows(source_rows).items():
                    metrics = summarize_parse_source(group_rows)
                    metric_rows.append(
                        {
                            "model_id": model_id,
                            "source_group": group_name,
                            "parser_source": source_name,
                            **{k: v for k, v in metrics.items() if k != "parse_status_counts"},
                        }
                    )
                    for status, count in sorted(metrics["parse_status_counts"].items()):
                        parse_compare_rows.append(
                            {
                                "dataset": "stageD",
                                "model_id": model_id,
                                "source_group": group_name,
                                "parser_source": source_name,
                                "parse_status": status,
                                "count": int(count),
                            }
                        )

            model_summaries[model_id] = {
                "predictions_parquet": str(pred_path),
                "processed_rows": int(len(rows)),
                "flip_counts_historical_vs_new": dict(flip_counts),
            }

    metrics_path = out_dir / "stageD_metric_compat.csv"
    write_csv(
        metrics_path,
        metric_rows,
        [
            "model_id",
            "source_group",
            "parser_source",
            "n",
            "correct_count",
            "accuracy",
            "valid_answer_count",
            "coverage",
            "selective_accuracy",
            "abstain_count",
            "abstain_rate",
            "boxed_valid_count",
            "boxed_valid_rate",
            "parse_fail_count",
            "parse_fail_rate",
            "valid_non_abstain_wrong_count",
            "answered_error_rate",
        ],
    )
    parse_compare_path = out_dir / "stageD_parse_status_compare.csv"
    write_csv(
        parse_compare_path,
        parse_compare_rows,
        ["dataset", "model_id", "source_group", "parser_source", "parse_status", "count"],
    )

    return {
        "kind": "stageD",
        "eval_root": str(args.stageD_eval_root),
        "model_ids": model_ids,
        "max_stageD_rows": int(args.max_stageD_rows),
        "models": model_summaries,
        "outputs": {
            "stageD_metric_compat_csv": str(metrics_path),
            "stageD_parse_status_compare_csv": str(parse_compare_path),
            "stageD_response_flip_examples_jsonl": str(examples_path),
        },
    }


def write_combined_parse_status(out_dir: Path, kbp_summary: dict[str, Any] | None) -> str:
    rows: list[dict[str, Any]] = []
    if kbp_summary:
        counts_by_source = kbp_summary.get("parse_status_counts", {})
        all_statuses: set[str] = set()
        for counts in counts_by_source.values():
            all_statuses.update(str(k) for k in counts.keys())
        for status in sorted(all_statuses):
            row = {"dataset": "kbp", "parse_status": status}
            for source_name, counts in counts_by_source.items():
                row[source_name] = int(counts.get(status, 0))
            rows.append(row)

    path = out_dir / "parse_status_compare.csv"
    write_csv(
        path,
        rows,
        [
            "dataset",
            "parse_status",
            "historical_persisted",
            "old_compat_final_answer_last_boxed",
            "new_final_answer_last_boxed",
        ],
    )
    return str(path)


def write_legacy_first_boxed_probe(parser_module: Any, out_dir: Path) -> str:
    cases = [
        ("single_choice", r"\boxed{A}"),
        ("aspirin_word", r"\boxed{Aspirin}"),
        ("option_with_text", r"\boxed{A. Aspirin}"),
        ("multi_boxed", r"\boxed{B} then \boxed{A}"),
        ("answer_block_last", r"<think>\boxed{B}</think><answer>final \boxed{C}</answer>"),
        ("nested_text", r"\boxed{\text{A}}"),
        ("nested_choice_text", r"\boxed{A \text{because}}"),
        ("no_boxed", "no boxed answer"),
    ]
    rows: list[dict[str, Any]] = []
    target_set = {"A"}
    choice_set = {"A", "B", "C", "D"}
    for name, text in cases:
        old = old_parse_first(text, target_set, choice_set)
        new = new_parse(
            parser_module,
            text,
            policy=POLICY_LEGACY_FIRST,
            target_set=target_set,
            choice_set=choice_set,
        )
        rows.append(
            {
                "case": name,
                "old_boxed_raw": old["boxed_raw"],
                "old_normalized_choice": old["normalized_choice"],
                "old_parse_status": old["parse_status"],
                "new_boxed_raw": new["boxed_raw"],
                "new_normalized_choice": new["normalized_choice"],
                "new_parse_status": new["parse_status"],
                "flip_category": flip_category(old, new),
            }
        )
    path = out_dir / "legacy_first_boxed_probe.csv"
    write_csv(
        path,
        rows,
        [
            "case",
            "old_boxed_raw",
            "old_normalized_choice",
            "old_parse_status",
            "new_boxed_raw",
            "new_normalized_choice",
            "new_parse_status",
            "flip_category",
        ],
    )
    return str(path)


def write_summary_md(
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    kbp_summary: dict[str, Any] | None,
    stageD_summary: dict[str, Any] | None,
) -> str:
    lines = [
        "# Phase C Parser Replay Compatibility",
        "",
        "This report is an offline parser/metric replay. It did not train models, generate responses, or modify historical outputs.",
        "",
        "## Run",
        "",
        f"- generated_at_utc: `{metadata['generated_at_utc']}`",
        f"- truthrl_commit: `{metadata.get('truthrl_commit', '')}`",
        f"- out_dir: `{out_dir}`",
        f"- smoke_mode: `{metadata['smoke_mode']}`",
        "",
    ]

    if kbp_summary:
        q_changes = kbp_summary.get("question_change_counts", {})
        lines.extend(
            [
                "## KBP",
                "",
                f"- run_dir: `{kbp_summary['run_dir']}`",
                f"- processed_responses: `{kbp_summary['processed_responses']}`",
                f"- question_count: `{kbp_summary['question_count']}`",
                f"- labels_match_historical_counts: `{kbp_summary['labels_match_historical_counts']}`",
                f"- flip_counts_historical_vs_new: `{json.dumps(kbp_summary.get('flip_counts', {}), sort_keys=True)}`",
                f"- question_change_counts: `{json.dumps(q_changes, sort_keys=True)}`",
                f"- special_counts: `{json.dumps(kbp_summary.get('special_counts', {}), sort_keys=True)}`",
                "",
            ]
        )

    if stageD_summary:
        lines.extend(["## StageD", ""])
        for model_id, model_payload in stageD_summary.get("models", {}).items():
            lines.append(f"- `{model_id}` processed_rows: `{model_payload['processed_rows']}`")
            lines.append(
                f"- `{model_id}` flip_counts_historical_vs_new: "
                f"`{json.dumps(model_payload.get('flip_counts_historical_vs_new', {}), sort_keys=True)}`"
            )
        lines.append("")

    lines.extend(
        [
            "## Outputs",
            "",
        ]
    )
    for key, value in metadata.get("outputs", {}).items():
        lines.append(f"- {key}: `{value}`")

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    smoke_mode = args.max_kbp_responses > 0 or args.max_stageD_rows > 0
    out_dir = args.out_dir or default_out_dir(args.repo_root, smoke_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    parser_module = load_parser_module(args.repo_root)
    kbp_summary = None if args.skip_kbp else replay_kbp(args=args, parser_module=parser_module, out_dir=out_dir)
    stageD_summary = None if args.skip_stageD else replay_stageD(args=args, parser_module=parser_module, out_dir=out_dir)

    parse_status_path = write_combined_parse_status(out_dir, kbp_summary)
    legacy_probe_path = write_legacy_first_boxed_probe(parser_module, out_dir)

    metadata = {
        "generated_at_utc": utc_now(),
        "repo_root": str(args.repo_root),
        "truthrl_commit": git_head(args.repo_root),
        "smoke_mode": bool(smoke_mode),
        "max_kbp_responses": int(args.max_kbp_responses),
        "max_stageD_rows": int(args.max_stageD_rows),
        "parser_module": str(
            args.repo_root / "training" / "verl" / "verl" / "utils" / "reward_score" / "medical_answer_parser.py"
        ),
        "policies": {
            "kbp_new": POLICY_KBP_NEW,
            "stageD_new": POLICY_STAGE_D_NEW,
            "legacy_probe_new": POLICY_LEGACY_FIRST,
        },
        "outputs": {
            "parse_status_compare_csv": parse_status_path,
            "legacy_first_boxed_probe_csv": legacy_probe_path,
        },
    }
    if kbp_summary:
        metadata["kbp"] = kbp_summary
        metadata["outputs"].update(kbp_summary["outputs"])
    if stageD_summary:
        metadata["stageD"] = stageD_summary
        metadata["outputs"].update(stageD_summary["outputs"])

    summary_path = write_summary_md(
        out_dir=out_dir,
        metadata=metadata,
        kbp_summary=kbp_summary,
        stageD_summary=stageD_summary,
    )
    metadata["outputs"]["summary_md"] = summary_path
    metadata_path = out_dir / "metadata.json"
    write_json(metadata_path, metadata)

    print(f"[PASS] wrote Phase C parser replay outputs under: {out_dir}")
    print(f"[INFO] summary={summary_path}")
    print(f"[INFO] metadata={metadata_path}")


if __name__ == "__main__":
    main()
