#!/usr/bin/env python3
"""Offline reward replay utility for Stage D."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Stage-D reward on parquet samples.")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--reward_fn_path", type=Path, required=True)
    parser.add_argument("--reward_fn_name", type=str, default="compute_score")
    parser.add_argument("--prediction_jsonl", type=Path, default=None)
    parser.add_argument("--synthetic_policy", type=str, default="triad", choices=["triad", "all_correct"])
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stage_mode", type=str, default="d1", choices=["d1", "d2"])
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--enable_format", action="store_true")
    parser.add_argument("--enable_consistency", action="store_true")
    parser.add_argument("--lambda_format", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=0.5)
    parser.add_argument("--out_json", type=Path, required=True)
    return parser.parse_args()


def _load_reward_fn(path: Path, fn_name: str):
    if not path.exists():
        raise FileNotFoundError(f"reward function file not found: {path}")
    spec = importlib.util.spec_from_file_location("stage_d_reward_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, fn_name):
        raise AttributeError(f"{fn_name} not found in {path}")
    return getattr(module, fn_name)


def _load_rows(parquet_path: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    table = pq.read_table(parquet_path, columns=["data_source", "reward_model", "extra_info"])
    rows = table.to_pylist()
    if len(rows) <= sample_size:
        return rows
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    return [rows[i] for i in idxs[:sample_size]]


def _load_predictions(pred_path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj["question_id"])
            preds[qid] = str(obj["response"])
    return preds


def _pick_wrong_choice(choices: list[str], target: str) -> str:
    labels: list[str] = []
    for item in choices:
        if not isinstance(item, str):
            continue
        prefix = item.split(":", 1)[0].strip().upper()
        if len(prefix) == 1 and "A" <= prefix <= "Z":
            labels.append(prefix)
    for c in labels:
        if c != target:
            return c
    return "Z"


def _synthetic_response(policy: str, idx: int, gt: dict[str, Any]) -> str:
    target_list = gt.get("target", [])
    target = target_list[0] if isinstance(target_list, list) and target_list else "A"
    if policy == "all_correct":
        return f"<think>synthetic</think><answer>\\boxed{{{target}}}</answer>"

    bucket = idx % 3
    if bucket == 0:
        return f"<think>synthetic</think><answer>\\boxed{{{target}}}</answer>"
    if bucket == 1:
        return "<think>synthetic</think><answer>\\boxed{I don't know}</answer>"
    wrong = _pick_wrong_choice(gt.get("choices", []), str(target).strip().upper())
    return f"<think>synthetic</think><answer>\\boxed{{{wrong}}}</answer>"


def main() -> None:
    args = parse_args()

    reward_fn = _load_reward_fn(args.reward_fn_path, args.reward_fn_name)
    rows = _load_rows(args.parquet, args.sample_size, args.seed)
    pred_map = _load_predictions(args.prediction_jsonl) if args.prediction_jsonl else {}

    score_values: list[float] = []
    key_counters: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_sums: dict[str, float] = defaultdict(float)
    numeric_counts: dict[str, int] = defaultdict(int)
    missing_pred = 0

    for i, row in enumerate(rows):
        data_source = row.get("data_source", "unknown")
        reward_model = row.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", {})
        extra_info = row.get("extra_info", {})
        qid = str(extra_info.get("question_id", f"row_{i}"))

        if qid in pred_map:
            response = pred_map[qid]
        elif pred_map:
            missing_pred += 1
            continue
        else:
            response = _synthetic_response(args.synthetic_policy, i, ground_truth)

        result = reward_fn(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
            stage_mode=args.stage_mode,
            k=args.k,
            enable_format=args.enable_format,
            enable_consistency=args.enable_consistency,
            lambda_format=args.lambda_format,
            lambda_consistency=args.lambda_consistency,
        )

        if isinstance(result, dict):
            score = float(result.get("score", 0.0))
            score_values.append(score)
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    numeric_sums[k] += float(v)
                    numeric_counts[k] += 1
                elif isinstance(v, str):
                    key_counters[k][v] += 1
        else:
            score_values.append(float(result))

    n = len(score_values)
    if n == 0:
        raise RuntimeError("no scored samples available for replay")

    score_counter = Counter(round(x, 4) for x in score_values)
    numeric_means = {
        k: (numeric_sums[k] / numeric_counts[k]) for k in sorted(numeric_sums) if numeric_counts[k] > 0
    }
    str_distributions = {k: dict(v) for k, v in key_counters.items()}

    report = {
        "parquet": str(args.parquet),
        "reward_fn_path": str(args.reward_fn_path),
        "reward_fn_name": args.reward_fn_name,
        "sample_size_requested": args.sample_size,
        "num_scored": n,
        "num_missing_prediction": missing_pred,
        "stage_mode": args.stage_mode,
        "reward_kwargs": {
            "k": args.k,
            "enable_format": args.enable_format,
            "enable_consistency": args.enable_consistency,
            "lambda_format": args.lambda_format,
            "lambda_consistency": args.lambda_consistency,
        },
        "score_mean": sum(score_values) / n,
        "score_min": min(score_values),
        "score_max": max(score_values),
        "score_histogram": dict(sorted(score_counter.items(), key=lambda x: x[0])),
        "numeric_metric_mean": numeric_means,
        "string_metric_distribution": str_distributions,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote replay report: {args.out_json}")


if __name__ == "__main__":
    main()
