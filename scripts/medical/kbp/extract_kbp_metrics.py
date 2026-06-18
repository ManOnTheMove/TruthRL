#!/usr/bin/env python3
"""Extract new-parser KBP metrics for visualization.

Versioned from the non-git support script:

    /project/def-zshakeri/erichyu/embc/KBP_results/scripts/extract_kbp_metrics.py

This TruthRL copy recomputes KBP metrics with the shared parser under the
``final_answer_last_boxed`` policy. It writes regenerated metrics under an
explicit ``new_parser`` namespace and does not overwrite historical KBP labels
or persisted historical ``correct_count`` / ``difficulty_bucket`` fields.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_KBP_ROOT = Path("/project/def-zshakeri/erichyu/embc/TruthRL/data/medical/kbp")
DEFAULT_CANONICAL_RUN_ID = "kbp_20260319_stagec-c8-8b_medqa_grpo_train_fullpass"
DEFAULT_INPUT_PARQUET = Path("/project/def-zshakeri/erichyu/embc/TruthRL/data/medical/verl/medqa_grpo_train.parquet")

BUCKET_ORDER = ["0", "1-63", "64-127", "128-191", "192-255", "256", "other"]
ORIGINAL_NON_GIT_SCRIPT = "/project/def-zshakeri/erichyu/embc/KBP_results/scripts/extract_kbp_metrics.py"
METRIC_NAMESPACE = "new_parser_final_answer_last_boxed"
PARSER_POLICY = "final_answer_last_boxed"
PARSE_POLICY = PARSER_POLICY
PARSER_MODULE = "verl.utils.reward_score.medical_answer_parser"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract KBP metrics to CSV files.")
    parser.add_argument("--kbp_root", type=Path, default=DEFAULT_KBP_ROOT)
    parser.add_argument("--canonical_run_id", type=str, default=DEFAULT_CANONICAL_RUN_ID)
    parser.add_argument("--truthrl_root", type=Path, default=None)
    parser.add_argument("--default_input_parquet", type=Path, default=DEFAULT_INPUT_PARQUET)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head(truthrl_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(truthrl_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def difficulty_bucket(correct_count: int) -> str:
    c = int(correct_count)
    if c == 0:
        return "0"
    if 1 <= c <= 63:
        return "1-63"
    if 64 <= c <= 127:
        return "64-127"
    if 128 <= c <= 191:
        return "128-191"
    if 192 <= c <= 255:
        return "192-255"
    if c == 256:
        return "256"
    return "other"


def discover_run_summaries(kbp_root: Path) -> list[tuple[str, Path]]:
    runs_dir = kbp_root / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"KBP runs directory not found: {runs_dir}")

    out: list[tuple[str, Path]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "reports" / "kbp_summary.json"
        if summary_path.exists():
            out.append((run_dir.name, summary_path))
    if not out:
        raise RuntimeError(f"No KBP summary files found under: {runs_dir}")
    return out


def infer_truthrl_root(kbp_root: Path) -> Path:
    candidates = [
        kbp_root.parents[2],
        Path("/project/def-zshakeri/erichyu/embc/TruthRL"),
        Path("/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL"),
    ]
    for cand in candidates:
        reward_path = cand / "training" / "verl" / "verl" / "utils" / "reward_score" / "clinical_medqa_reward.py"
        if reward_path.exists():
            return cand
    raise FileNotFoundError(f"Unable to infer TruthRL root from kbp_root={kbp_root}")


def load_reward_module(truthrl_root: Path) -> Any:
    reward_path = truthrl_root / "training" / "verl" / "verl" / "utils" / "reward_score" / "clinical_medqa_reward.py"
    if not reward_path.exists():
        raise FileNotFoundError(f"Reward module not found: {reward_path}")

    truthrl_verl = str((truthrl_root / "training" / "verl").resolve())
    if sys.path[0] != truthrl_verl:
        sys.path.insert(0, truthrl_verl)

    spec = importlib.util.spec_from_file_location("clinical_medqa_reward", str(reward_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reward module from: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_existing_input_parquet(path_like: Any, default_input_parquet: Path) -> Path:
    candidates: list[Path] = []

    if path_like:
        p = Path(str(path_like))
        candidates.append(p)
        s = str(p)
        legacy_prefixes = [
            "/home/erichyu/projects/def-zshakeri/erichyu/",
            "/home/erichyu/links/projects/def-zshakeri/erichyu/",
        ]
        for prefix in legacy_prefixes:
            if s.startswith(prefix):
                candidates.append(Path("/project/def-zshakeri/erichyu/" + s[len(prefix) :]))

    candidates.append(default_input_parquet)

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot resolve input parquet from path={path_like}, default={default_input_parquet}")


def _extract_target_set(reward_module: Any, ground_truth: dict[str, Any]) -> set[str]:
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    out: set[str] = set()
    if isinstance(targets, list):
        for t in targets:
            n = reward_module.normalize_choice(str(t))
            if n:
                out.add(n)
    return out


def _extract_choice_set(reward_module: Any, ground_truth: dict[str, Any]) -> set[str]:
    choices = ground_truth.get("choices", [])
    out: set[str] = set()
    if isinstance(choices, list):
        for c in choices:
            if not isinstance(c, str):
                continue
            n = reward_module.normalize_choice(c)
            if n:
                out.add(n)
                continue
            if ":" in c:
                n2 = reward_module.normalize_choice(c.split(":", 1)[0])
                if n2:
                    out.add(n2)
    return out


def load_target_choice_maps(
    input_parquet: Path,
    reward_module: Any,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    table = pq.read_table(input_parquet, columns=["reward_model", "extra_info"])
    rows = table.to_pylist()
    target_map: dict[str, set[str]] = {}
    choice_map: dict[str, set[str]] = {}
    for idx, row in enumerate(rows):
        extra_info = row.get("extra_info") or {}
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") or {}
        qid = str(extra_info.get("question_id", f"row_{idx}"))
        target_map[qid] = _extract_target_set(reward_module, ground_truth)
        choice_map[qid] = _extract_choice_set(reward_module, ground_truth)
    return target_map, choice_map


def load_run_question_ids(run_dir: Path) -> list[str]:
    prompts_path = run_dir / "prompts.parquet"
    if not prompts_path.exists():
        return []
    table = pq.read_table(prompts_path, columns=["question_id"])
    qids = [str(x) for x in table.column("question_id").to_pylist() if x is not None]
    return sorted(set(qids))


def iter_response_parts(run_dir: Path, split: str) -> list[Path]:
    preferred = run_dir / "responses" / f"split={split}"
    if preferred.exists():
        return sorted(preferred.glob("part-*.parquet"))

    responses_dir = run_dir / "responses"
    if not responses_dir.exists():
        return []
    split_dirs = sorted([p for p in responses_dir.iterdir() if p.is_dir() and p.name.startswith("split=")])
    if not split_dirs:
        return []
    return sorted(split_dirs[0].glob("part-*.parquet"))


def extract_final_boxed(reward_module: Any, response_text: str) -> tuple[bool, str]:
    parsed = reward_module.parse_medical_answer(
        response_text,
        policy=reward_module.POLICY_FINAL_ANSWER_LAST_BOXED,
    )
    if parsed.parse_status == "no_boxed":
        return False, ""
    return True, parsed.boxed_raw


def parse_response(
    reward_module: Any,
    response_text: str,
    target_set: set[str],
    choice_set: set[str],
) -> dict[str, Any]:
    parsed = reward_module.parse_medical_answer(
        response_text,
        policy=reward_module.POLICY_FINAL_ANSWER_LAST_BOXED,
        choice_set=choice_set,
    )
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
    }


def recompute_run_from_responses(
    run_dir: Path,
    split: str,
    reward_module: Any,
    target_map: dict[str, set[str]],
    choice_map: dict[str, set[str]],
) -> dict[str, Any]:
    question_ids = load_run_question_ids(run_dir)
    probe_count: dict[str, int] = {qid: 0 for qid in question_ids}
    correct_count: dict[str, int] = {qid: 0 for qid in question_ids}
    abstain_count: dict[str, int] = {qid: 0 for qid in question_ids}
    parse_status_counter: Counter[str] = Counter()
    responses_row_count = 0

    part_paths = iter_response_parts(run_dir, split=split)
    for path in part_paths:
        table = pq.ParquetFile(path).read(columns=["question_id", "response_text"])
        qids = table.column("question_id").to_pylist()
        texts = table.column("response_text").to_pylist()
        responses_row_count += len(qids)

        for qid_raw, response_text in zip(qids, texts):
            qid = str(qid_raw) if qid_raw is not None else ""
            parsed = parse_response(
                reward_module=reward_module,
                response_text=response_text if isinstance(response_text, str) else "",
                target_set=target_map.get(qid, set()),
                choice_set=choice_map.get(qid, set()),
            )
            parse_status_counter[str(parsed["parse_status"])] += 1

            if not qid:
                continue
            if qid not in probe_count:
                probe_count[qid] = 0
                correct_count[qid] = 0
                abstain_count[qid] = 0
            probe_count[qid] += 1
            correct_count[qid] += int(parsed["is_correct"])
            abstain_count[qid] += int(parsed["is_abstain"])

    question_rows: list[dict[str, Any]] = []
    for qid in sorted(probe_count.keys()):
        p = int(probe_count.get(qid, 0))
        c = int(correct_count.get(qid, 0))
        a = int(abstain_count.get(qid, 0))
        question_rows.append(
            {
                "question_id": qid,
                "probe_count": p,
                "new_parser_correct_count": c,
                "new_parser_abstain_count": a,
                "new_parser_ook": bool(c == 0),
                "new_parser_correct_rate": float(c) / float(p if p > 0 else 1),
                "parser_namespace": METRIC_NAMESPACE,
                "parser_policy": PARSER_POLICY,
                "parser_module": PARSER_MODULE,
            }
        )

    question_df = pd.DataFrame(question_rows)
    if not question_df.empty:
        question_df["new_parser_difficulty_bucket"] = question_df["new_parser_correct_count"].map(difficulty_bucket)
        question_df = question_df.sort_values(by="question_id").reset_index(drop=True)
    else:
        question_df = pd.DataFrame(
            columns=[
                "question_id",
                "probe_count",
                "new_parser_correct_count",
                "new_parser_abstain_count",
                "new_parser_ook",
                "new_parser_correct_rate",
                "new_parser_difficulty_bucket",
                "parser_namespace",
                "parser_policy",
                "parser_module",
            ]
        )

    return {
        "responses_row_count": int(responses_row_count),
        "parse_status_counts": dict(parse_status_counter),
        "question_df": question_df,
        "question_count": int(len(question_df)),
        "correct_total": int(question_df["new_parser_correct_count"].sum()) if not question_df.empty else 0,
        "abstain_total": int(question_df["new_parser_abstain_count"].sum()) if not question_df.empty else 0,
        "ook_question_count": int(question_df["new_parser_ook"].astype(bool).sum()) if not question_df.empty else 0,
    }


def build_run_level_and_parse_rows(
    discovered: list[tuple[str, Path]],
    reward_module: Any,
    default_input_parquet: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, pd.DataFrame], dict[str, str]]:
    run_rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []
    run_question_df: dict[str, pd.DataFrame] = {}
    run_input_parquet: dict[str, str] = {}
    input_cache: dict[str, tuple[dict[str, set[str]], dict[str, set[str]]]] = {}

    for run_dir_name, summary_path in discovered:
        summary = read_json(summary_path)
        run_dir = summary_path.parents[1]
        validation = summary.get("validation") or {}
        checks = validation.get("checks") or {}
        run_id = str(summary.get("run_id") or run_dir_name)
        split = str(summary.get("split") or "medqa_grpo_train")
        input_parquet = _resolve_existing_input_parquet(summary.get("input_parquet"), default_input_parquet)
        input_key = str(input_parquet)
        run_input_parquet[run_id] = input_key

        if input_key not in input_cache:
            input_cache[input_key] = load_target_choice_maps(input_parquet=input_parquet, reward_module=reward_module)
        target_map, choice_map = input_cache[input_key]

        recomputed = recompute_run_from_responses(
            run_dir=run_dir,
            split=split,
            reward_module=reward_module,
            target_map=target_map,
            choice_map=choice_map,
        )
        run_question_df[run_id] = recomputed["question_df"]

        responses_row_count = int(recomputed["responses_row_count"])
        expected_row_count = _to_int(validation.get("expected_row_count"))

        run_rows.append(
            {
                "run_id": run_id,
                "run_dir_name": run_dir_name,
                "stage": str(summary.get("stage") or ""),
                "status": str(summary.get("status") or ""),
                "split": split,
                "question_count": int(recomputed["question_count"]) or _to_int(summary.get("question_count")),
                "responses_row_count": responses_row_count,
                "expected_row_count": expected_row_count,
                "row_count_gap": responses_row_count - expected_row_count,
                "ook_question_count": int(recomputed["ook_question_count"]),
                "correct_total": int(recomputed["correct_total"]),
                "abstain_total": int(recomputed["abstain_total"]),
                "new_parser_ook_question_count": int(recomputed["ook_question_count"]),
                "new_parser_correct_total": int(recomputed["correct_total"]),
                "new_parser_abstain_total": int(recomputed["abstain_total"]),
                "overall_pass": _to_bool(validation.get("overall_pass")),
                "row_count_ok": _to_bool(checks.get("row_count_ok")),
                "probe_count_per_question_ok": _to_bool(checks.get("probe_count_per_question_ok")),
                "probe_index_coverage_ok": _to_bool(checks.get("probe_index_coverage_ok")),
                "empty_response_ok": _to_bool(checks.get("empty_response_ok")),
                "parsed_matches_responses": _to_bool(checks.get("parsed_matches_responses")),
                "summary_recomputable_ok": _to_bool(checks.get("summary_recomputable_ok")),
                "probe_count_min": _to_int(validation.get("probe_count_min")),
                "probe_count_max": _to_int(validation.get("probe_count_max")),
                "empty_response_count": _to_int(validation.get("empty_response_count")),
                "empty_response_ratio": _to_float(validation.get("empty_response_ratio")),
                "started_at_utc": str(summary.get("started_at_utc") or ""),
                "ended_at_utc": str(summary.get("ended_at_utc") or ""),
                "duration_seconds": _to_float(summary.get("duration_seconds")),
                "summary_path": str(summary_path),
                "metric_namespace": METRIC_NAMESPACE,
                "parse_policy": PARSER_POLICY,
                "parser_policy": PARSER_POLICY,
                "parser_module": PARSER_MODULE,
            }
        )

        parse_status_counts = recomputed["parse_status_counts"] or {}
        total_for_ratio = responses_row_count if responses_row_count > 0 else 1
        for parse_status, count in sorted(parse_status_counts.items()):
            c = _to_int(count)
            parse_rows.append(
                {
                    "run_id": run_id,
                    "parse_status": str(parse_status),
                    "count": c,
                    "responses_row_count": responses_row_count,
                    "ratio": float(c) / float(total_for_ratio),
                    "metric_namespace": METRIC_NAMESPACE,
                    "parse_policy": PARSER_POLICY,
                    "parser_policy": PARSER_POLICY,
                    "parser_module": PARSER_MODULE,
                }
            )

    return run_rows, parse_rows, run_question_df, run_input_parquet


def build_bucket_metrics(question_df: pd.DataFrame) -> pd.DataFrame:
    total = len(question_df)
    grouped = (
        question_df.groupby("new_parser_difficulty_bucket", dropna=False)
        .size()
        .reset_index(name="question_count")
    )
    grouped["ratio"] = grouped["question_count"] / float(total if total > 0 else 1)
    grouped["metric_namespace"] = METRIC_NAMESPACE
    grouped["parser_policy"] = PARSER_POLICY
    grouped["parser_module"] = PARSER_MODULE
    grouped["bucket_order"] = grouped["new_parser_difficulty_bucket"].map(
        {b: i for i, b in enumerate(BUCKET_ORDER)}
    ).fillna(999)

    return grouped.sort_values(by=["bucket_order", "new_parser_difficulty_bucket"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    truthrl_root = args.truthrl_root if args.truthrl_root is not None else infer_truthrl_root(args.kbp_root)
    reward_module = load_reward_module(truthrl_root)
    code_commit = git_head(truthrl_root)

    discovered = discover_run_summaries(args.kbp_root)
    run_rows, parse_rows, run_question_df, run_input_parquet = build_run_level_and_parse_rows(
        discovered=discovered,
        reward_module=reward_module,
        default_input_parquet=args.default_input_parquet,
    )

    run_df = pd.DataFrame(run_rows)
    run_df = run_df.sort_values(by=["run_id"]).reset_index(drop=True)
    run_level_path = out_dir / "run_level_metrics.csv"
    run_df.to_csv(run_level_path, index=False)

    parse_df = pd.DataFrame(parse_rows)
    parse_df = parse_df.sort_values(by=["run_id", "parse_status"]).reset_index(drop=True)
    run_parse_path = out_dir / "run_parse_status.csv"
    parse_df.to_csv(run_parse_path, index=False)

    if args.canonical_run_id not in set(run_df["run_id"].astype(str)):
        raise RuntimeError(
            f"canonical_run_id={args.canonical_run_id} not found in discovered runs. "
            f"Available: {sorted(set(run_df['run_id'].astype(str)))}"
        )

    q_df = run_question_df[args.canonical_run_id].copy()
    if q_df.empty:
        raise RuntimeError(f"canonical_run_id={args.canonical_run_id} has empty recomputed question metrics")

    question_path = out_dir / "canonical_question_metrics.csv"
    q_df.to_csv(question_path, index=False)

    bucket_df = build_bucket_metrics(q_df)
    bucket_path = out_dir / "canonical_bucket_metrics.csv"
    bucket_df.to_csv(bucket_path, index=False)

    metadata = {
        "generated_at_utc": utc_now(),
        "code_commit": code_commit,
        "kbp_root": str(args.kbp_root),
        "truthrl_root": str(truthrl_root),
        "canonical_run_id": args.canonical_run_id,
        "input_paths": {
            "kbp_root": str(args.kbp_root),
            "canonical_input_parquet": run_input_parquet.get(args.canonical_run_id, ""),
            "default_input_parquet": str(args.default_input_parquet),
        },
        "output_dir": str(out_dir),
        "metric_namespace": METRIC_NAMESPACE,
        "parser_policy": PARSER_POLICY,
        "parse_policy": PARSER_POLICY,
        "parser_module": PARSER_MODULE,
        "original_non_git_script": ORIGINAL_NON_GIT_SCRIPT,
        "historical_metric_policy": (
            "Historical KBP correct_count and difficulty_bucket fields must not be overwritten. "
            "This script emits regenerated values with new_parser_* column names."
        ),
        "discovered_run_count": int(len(discovered)),
        "canonical_input_parquet": run_input_parquet.get(args.canonical_run_id, ""),
        "outputs": {
            "run_level_metrics_csv": str(run_level_path),
            "run_parse_status_csv": str(run_parse_path),
            "canonical_question_metrics_csv": str(question_path),
            "canonical_bucket_metrics_csv": str(bucket_path),
        },
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[PASS] wrote {run_level_path}")
    print(f"[PASS] wrote {run_parse_path}")
    print(f"[PASS] wrote {question_path}")
    print(f"[PASS] wrote {bucket_path}")
    print(f"[PASS] wrote {metadata_path}")


if __name__ == "__main__":
    main()
