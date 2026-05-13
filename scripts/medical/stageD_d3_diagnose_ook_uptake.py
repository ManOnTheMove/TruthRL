#!/usr/bin/env python3
"""Diagnose hard-OOK signal uptake for Stage D D3.

This script is read-only with respect to training assets. It summarizes:
- D2.6 hard-OOK prevalence and prompt contract in the train parquet.
- Expected OOK exposure under the natural shuffled sampler.
- D3 training log signals for OOK and abstention.
- Phase 2/3 decoded eval behavior on D2.6 hard-OOK rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


STEP_PATTERN = re.compile(r"step:(\d+)\s+-\s+(.*)")
METRIC_PATTERN = re.compile(r"([A-Za-z0-9_@./-]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("reward_model") or {}).get("ground_truth") or {}


def prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("prompt") or []
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
        else:
            parts.append(str(message))
    return "\n".join(parts)


def question_id(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    return str(extra_info.get("question_id") or extra_info.get("index") or "")


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "median": statistics.median(ordered),
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def summarize_dataset(train_path: Path, val_path: Path, train_batch_size: int, rollout_n: int, steps: int) -> dict[str, Any]:
    train_rows = read_parquet_rows(train_path)
    val_rows = read_parquet_rows(val_path)

    ook_rows = [(i, row) for i, row in enumerate(train_rows) if bool(ground_truth(row).get("out_of_knowledge", False))]
    known_rows = [(i, row) for i, row in enumerate(train_rows) if not bool(ground_truth(row).get("out_of_knowledge", False))]
    val_ook = sum(bool(ground_truth(row).get("out_of_knowledge", False)) for row in val_rows)

    def prompt_contract(subset: list[tuple[int, dict[str, Any]]]) -> dict[str, int]:
        texts = [prompt_text(row) for _, row in subset]
        return {
            "contains_i_dont_know": sum("I don't know" in text or "I don’t know" in text for text in texts),
            "contains_do_not_know": sum("do not know" in text.lower() for text in texts),
            "contains_boxed_i_dont_know": sum("\\boxed{I don't know}" in text for text in texts),
        }

    n = len(train_rows)
    k = len(ook_rows)
    p = k / n if n else 0.0
    p_zero_hypergeom = 1.0
    for j in range(min(train_batch_size, n - k)):
        p_zero_hypergeom *= (n - k - j) / (n - j)

    first_ook = ook_rows[0] if ook_rows else None
    first_ook_payload = None
    if first_ook:
        idx, row = first_ook
        first_ook_payload = {
            "row_index": idx,
            "question_id": question_id(row),
            "ground_truth": ground_truth(row),
            "prompt_tail": prompt_text(row)[-700:].replace("\n", " "),
        }

    return {
        "train_rows": n,
        "train_ook_rows": k,
        "train_known_rows": len(known_rows),
        "train_ook_rate": p,
        "val_rows": len(val_rows),
        "val_ook_rows": val_ook,
        "prompt_contract": {
            "ook": prompt_contract(ook_rows),
            "known": prompt_contract(known_rows),
            "val": prompt_contract(list(enumerate(val_rows))),
        },
        "ook_row_index_first20": [idx for idx, _ in ook_rows[:20]],
        "ook_row_index_last10": [idx for idx, _ in ook_rows[-10:]],
        "ook_target_distribution": {str(k): v for k, v in Counter(tuple(ground_truth(row).get("target", [])) for _, row in ook_rows).items()},
        "known_target_distribution_top10": [
            {"target": str(target), "count": count}
            for target, count in Counter(tuple(ground_truth(row).get("target", [])) for _, row in known_rows).most_common(10)
        ],
        "expected_exposure": {
            "train_batch_size": train_batch_size,
            "rollout_n": rollout_n,
            "steps": steps,
            "expected_ook_prompts_per_batch": train_batch_size * p,
            "expected_ook_rollouts_per_step": train_batch_size * rollout_n * p,
            "p_zero_ook_prompts_per_batch_binomial": (1 - p) ** train_batch_size if n else None,
            "p_zero_ook_prompts_per_batch_hypergeom": p_zero_hypergeom,
            "expected_ook_prompts_300_steps": steps * train_batch_size * p,
            "expected_ook_rollouts_300_steps": steps * train_batch_size * rollout_n * p,
        },
        "prompt_chars": {
            "ook": {
                "mean": mean([float(len(prompt_text(row))) for _, row in ook_rows]),
                **quantiles([float(len(prompt_text(row))) for _, row in ook_rows]),
            },
            "known": {
                "mean": mean([float(len(prompt_text(row))) for _, row in known_rows]),
                **quantiles([float(len(prompt_text(row))) for _, row in known_rows]),
            },
        },
        "first_ook": first_ook_payload,
    }


def parse_training_steps(log_path: Path, train_batch_size: int) -> dict[str, Any]:
    step_records: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = STEP_PATTERN.search(line)
        if not match:
            continue
        step = int(match.group(1))
        metrics = {key: float(value) for key, value in METRIC_PATTERN.findall(match.group(2))}
        metrics["step"] = float(step)
        step_records.append(metrics)

    keys = [
        "critic/rewards/is_out_of_knowledge",
        "critic/rewards/is_abstain",
        "critic/rewards/is_boxed_valid",
        "critic/rewards/mean",
        "critic/rewards/outcome_score",
        "val-aux/medqa/is_abstain/mean@1",
        "val-aux/medqa/is_out_of_knowledge/mean@1",
        "val-aux/medqa/is_boxed_valid/mean@1",
        "val-core/medqa/reward/mean@1",
    ]

    summary: dict[str, Any] = {"num_logged_steps": len(step_records)}
    for key in keys:
        values = [record[key] for record in step_records if key in record]
        summary[key] = {
            "count": len(values),
            "mean": mean(values),
            **quantiles(values),
        }

    ook_prompt_counts = [
        record["critic/rewards/is_out_of_knowledge"] * train_batch_size
        for record in step_records
        if "critic/rewards/is_out_of_knowledge" in record
    ]
    summary["approx_ook_prompts_per_logged_train_step"] = {
        "count": len(ook_prompt_counts),
        "mean": mean(ook_prompt_counts),
        **quantiles(ook_prompt_counts),
        "zero_steps": sum(1 for value in ook_prompt_counts if abs(value) < 1e-9),
    }
    summary["last_step"] = step_records[-1] if step_records else None
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize_predictions(predictions_path: Path) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    d26 = [row for row in rows if row.get("source_group") == "d26_hard_ook"]
    standard = [row for row in rows if row.get("source_group") == "standard_medqa"]

    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        return {
            "n": n,
            "abstain_count": sum(int(row.get("is_abstain", 0)) for row in subset),
            "abstain_rate": sum(int(row.get("is_abstain", 0)) for row in subset) / n if n else None,
            "boxed_valid_count": sum(int(row.get("is_boxed_valid", 0)) for row in subset),
            "boxed_valid_rate": sum(int(row.get("is_boxed_valid", 0)) for row in subset) / n if n else None,
            "correct_count": sum(int(row.get("is_correct", 0)) for row in subset),
            "correct_rate": sum(int(row.get("is_correct", 0)) for row in subset) / n if n else None,
            "parse_status_counts": dict(Counter(str(row.get("parse_status", "")) for row in subset)),
            "boxed_raw_counts_top20": Counter(str(row.get("boxed_raw", "")) for row in subset).most_common(20),
            "generated_tokens_mean": mean([float(row.get("generated_tokens", 0)) for row in subset if row.get("generated_tokens") is not None]),
        }

    d26_abstain = [row for row in d26 if int(row.get("is_abstain", 0))]
    d26_non_abstain = [row for row in d26 if not int(row.get("is_abstain", 0))]
    return {
        "predictions_path": str(predictions_path),
        "all_rows": len(rows),
        "standard": summarize_subset(standard),
        "d26_hard_ook": summarize_subset(d26),
        "d26_abstain_question_ids": [row.get("question_id") for row in d26_abstain],
        "d26_first_abstain_examples": [
            {
                "question_id": row.get("question_id"),
                "boxed_raw": row.get("boxed_raw"),
                "response_head": str(row.get("response_text", ""))[:300].replace("\n", " "),
            }
            for row in d26_abstain[:8]
        ],
        "d26_first_non_abstain_examples": [
            {
                "question_id": row.get("question_id"),
                "boxed_raw": row.get("boxed_raw"),
                "parse_status": row.get("parse_status"),
                "response_head": str(row.get("response_text", ""))[:300].replace("\n", " "),
            }
            for row in d26_non_abstain[:8]
        ],
    }


def read_comparison(comparison_csv: Path) -> list[dict[str, str]]:
    with comparison_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    dataset = payload["dataset"]
    train_log = payload["training_log"]
    phase23 = payload["phase23"]
    comparison = payload["comparison"]

    def comparison_value(group: str, metric: str, column: str) -> str:
        for row in comparison:
            if row["source_group"] == group and row["metric"] == metric:
                return row[column]
        return ""

    lines = [
        "# Stage D D3 Hard-OOK Uptake Diagnosis",
        "",
        "Date: 2026-05-11",
        "",
        "## Executive Summary",
        "",
        (
            "D3 的 hard-OOK abstention 失败最直接的解释是 hard-OOK 信号太稀疏，"
            "而训练配置没有任何 OOK-aware sampler / oversampling / group weighting。更关键的是，"
            "训练 prompt 中的 abstention option 并不是 OOK 专属信号：known train rows 也包含 "
            "`\\boxed{I don't know}` 选项，因此模型只能从极少数 OOK reward branch 中学习何时拒答。"
            f"训练集中 D2.6 OOK rows 只有 {dataset['train_ook_rows']} / {dataset['train_rows']} "
            f"({dataset['train_ook_rate']:.4%})。batch size 64、rollout_n 4 下，"
            f"每个训练 step 预期只有 {dataset['expected_exposure']['expected_ook_prompts_per_batch']:.2f} 个 OOK prompts，"
            f"约 {dataset['expected_exposure']['p_zero_ook_prompts_per_batch_hypergeom']:.2%} 的 batch 可能没有 OOK prompt。"
        ),
        "",
        (
            "Phase 2/3 decoded eval 已确认：D3 在 standard MedQA 上大幅提高 decoded accuracy 和 boxed validity，"
            "但在 D2.6 hard-OOK 上只拒答 5/98。训练日志也只记录 aggregate `critic/rewards/is_out_of_knowledge` "
            "和 `critic/rewards/is_abstain`，没有 OOK group-specific reward / abstain recall，因此 formal run "
            "没有足够监控来保证 OOK branch 被学到。"
        ),
        "",
        "## Data And Prompt Contract",
        "",
        f"- Train rows: `{dataset['train_rows']}`",
        f"- D2.6 hard-OOK rows: `{dataset['train_ook_rows']}`",
        f"- OOK rate: `{dataset['train_ook_rate']:.6f}`",
        f"- Validation rows: `{dataset['val_rows']}`",
        f"- Validation OOK rows: `{dataset['val_ook_rows']}`",
        (
            f"- OOK prompts containing `I don't know`: "
            f"`{dataset['prompt_contract']['ook']['contains_i_dont_know']} / {dataset['train_ook_rows']}`"
        ),
        (
            f"- Known train prompts containing `I don't know`: "
            f"`{dataset['prompt_contract']['known']['contains_i_dont_know']} / {dataset['train_known_rows']}`"
        ),
        (
            f"- Val prompts containing `I don't know`: "
            f"`{dataset['prompt_contract']['val']['contains_i_dont_know']} / {dataset['val_rows']}`"
        ),
        "",
        (
            "The prompt contract is not group-discriminative in train: both OOK and known training prompts include "
            "the abstention option. Validation prompts do not include the abstention option. This means the only "
            "training signal that distinguishes OOK from known is the reward label, and that signal is present in "
            "only 1.93% of train prompts."
        ),
        "",
        "## Natural Sampling Exposure",
        "",
        f"- Expected OOK prompts per 64-example batch: `{dataset['expected_exposure']['expected_ook_prompts_per_batch']:.3f}`",
        f"- Expected OOK rollouts per step with rollout_n=4: `{dataset['expected_exposure']['expected_ook_rollouts_per_step']:.3f}`",
        f"- Probability of zero OOK prompts per batch: `{dataset['expected_exposure']['p_zero_ook_prompts_per_batch_hypergeom']:.4f}`",
        f"- Expected OOK prompts over 300 steps: `{dataset['expected_exposure']['expected_ook_prompts_300_steps']:.1f}`",
        "",
        "The D3 formal run used ordinary shuffled sampling. There is no evidence of OOK oversampling or per-group reweighting.",
        "",
        "## Training Log Signals",
        "",
        f"- Logged step lines parsed: `{train_log['num_logged_steps']}`",
        (
            "- Approx OOK prompts per logged train step: "
            f"mean `{train_log['approx_ook_prompts_per_logged_train_step']['mean']:.3f}`, "
            f"median `{train_log['approx_ook_prompts_per_logged_train_step']['median']:.3f}`, "
            f"zero-step count `{train_log['approx_ook_prompts_per_logged_train_step']['zero_steps']}`"
        ),
        (
            "- Aggregate train `critic/rewards/is_abstain` mean over logged steps: "
            f"`{train_log['critic/rewards/is_abstain']['mean']:.6f}`"
        ),
        (
            "- Aggregate train `critic/rewards/is_out_of_knowledge` mean over logged steps: "
            f"`{train_log['critic/rewards/is_out_of_knowledge']['mean']:.6f}`"
        ),
        (
            "- Final validation OOK rate: "
            f"`{train_log['last_step'].get('val-aux/medqa/is_out_of_knowledge/mean@1') if train_log['last_step'] else None}`"
        ),
        (
            "- Final validation abstain rate: "
            f"`{train_log['last_step'].get('val-aux/medqa/is_abstain/mean@1') if train_log['last_step'] else None}`"
        ),
        "- Train rollout config from D3 log: `do_sample=True`, `temperature=1.0`, `n=4`.",
        "- Validation / Phase 2-3 eval decoding: greedy, `do_sample=False`, `temperature=0`.",
        "",
        (
            "Because the log metrics are aggregate, the OOK branch is diluted by the 98/5089 class ratio. "
            "Also, sampled train-time abstention is not equivalent to greedy checkpoint abstention. "
            "We cannot infer OOK recall from validation because validation OOK is 0."
        ),
        "",
        "## Phase 2/3 Decoded Evidence",
        "",
        "| Group | Metric | Stage C | D3 step300 | Delta |",
        "|---|---:|---:|---:|---:|",
        (
            f"| standard_medqa | total_accuracy | {float(comparison_value('standard_medqa', 'total_accuracy', 'stagec_base')):.4f} | "
            f"{float(comparison_value('standard_medqa', 'total_accuracy', 'd3_global_step_300')):.4f} | "
            f"{float(comparison_value('standard_medqa', 'total_accuracy', 'd3_global_step_300_minus_stagec_base')):.4f} |"
        ),
        (
            f"| standard_medqa | boxed_valid_rate | {float(comparison_value('standard_medqa', 'boxed_valid_rate', 'stagec_base')):.4f} | "
            f"{float(comparison_value('standard_medqa', 'boxed_valid_rate', 'd3_global_step_300')):.4f} | "
            f"{float(comparison_value('standard_medqa', 'boxed_valid_rate', 'd3_global_step_300_minus_stagec_base')):.4f} |"
        ),
        (
            f"| d26_hard_ook | abstain_rate | {float(comparison_value('d26_hard_ook', 'abstain_rate', 'stagec_base')):.4f} | "
            f"{float(comparison_value('d26_hard_ook', 'abstain_rate', 'd3_global_step_300')):.4f} | "
            f"{float(comparison_value('d26_hard_ook', 'abstain_rate', 'd3_global_step_300_minus_stagec_base')):.4f} |"
        ),
        (
            f"| d26_hard_ook | valid_non_abstain_wrong_rate | {float(comparison_value('d26_hard_ook', 'valid_non_abstain_wrong_rate', 'stagec_base')):.4f} | "
            f"{float(comparison_value('d26_hard_ook', 'valid_non_abstain_wrong_rate', 'd3_global_step_300')):.4f} | "
            f"{float(comparison_value('d26_hard_ook', 'valid_non_abstain_wrong_rate', 'd3_global_step_300_minus_stagec_base')):.4f} |"
        ),
        "",
        f"D3 D2.6 hard-OOK abstain question_ids: `{phase23['d26_abstain_question_ids']}`",
        "",
        "## Root-Cause Assessment",
        "",
        "Most likely causes, in priority order:",
        "",
        "1. OOK class imbalance: 1.93% of train prompts is likely too sparse for GRPO to learn a robust abstention policy.",
        "2. Train prompt is not group-discriminative: known and OOK train rows both include `I don't know`, so the prompt itself does not tell the model when abstention is expected.",
        "3. No group-aware sampling or weighting: D3 reward branch is correct, but OOK examples are not guaranteed to appear in every batch.",
        "4. Aggregate logging hides OOK failure: the run logs train-level `is_abstain` and `is_out_of_knowledge`, but not `P(abstain | OOK)` or OOK-group reward.",
        "5. Train/eval decoding mismatch: train rollouts are sampled at temperature 1.0, while Phase 2/3 and validation are greedy; sampled abstain events do not guarantee the learned greedy policy will abstain.",
        "6. Format learning dominates: D3 learned to emit valid boxed answers, which improves standard MedQA but converts prior parse failures on OOK rows into confident wrong answers.",
        "7. Validation set is non-OOK only: validation reward improvements could not select for OOK recall.",
        "",
        "Less likely from current evidence:",
        "",
        "- Abstention option missing from OOK rows. The OOK prompts do contain the `I don't know` abstention option.",
        "- Reward branch disabled. Config confirms `stage_mode=d3_hard_ook`, and logs include `critic/rewards/is_out_of_knowledge`.",
        "",
        "## Recommended Action",
        "",
        "Do not enter D4/CPRO yet. The next executable step should be D3.1 design, but only after adding a small CPU-side preflight and reward replay:",
        "",
        "1. Add OOK group-aware training diagnostics: per-step OOK prompt count, `P(abstain | OOK)`, OOK reward mean, known reward mean, and strong-known abstain false positive if available.",
        "2. Add an offline reward replay over D2.6 hard-OOK decoded outputs before any new training.",
        "3. For D3.1, use hard-OOK oversampling or group weighting so each batch contains a controlled OOK fraction, for example 10-20% prompts, while monitoring standard/strong-known over-abstention.",
        "4. Keep KBP v2 labels eval-only. They should not be written into training parquet or reward as hard labels.",
        "5. Run a short debug/smoke training only after the diagnostics above are in place; do not launch another long formal run blind.",
        "",
        "## Output References",
        "",
        f"- JSON diagnosis: `{payload['json_path']}`",
        f"- Phase 2/3 comparison: `{payload['comparison_csv']}`",
        f"- D3 predictions: `{payload['d3_predictions']}`",
        f"- Training log: `{payload['training_log_path']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, default=Path("/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/verl/medqa_grpo_train_with_ook_d26.parquet"))
    parser.add_argument("--val-parquet", type=Path, default=Path("/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/verl/medqa_grpo_test.parquet"))
    parser.add_argument("--training-log", type=Path, default=Path("/scratch/erichyu/stageD_d3_hard_ook/logs/stageD-d3-hard-ook-490120.out"))
    parser.add_argument("--comparison-csv", type=Path, default=Path("/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/phase23_20260511_standard_d26_stagec_vs_d3/comparison_by_group.csv"))
    parser.add_argument("--d3-predictions", type=Path, default=Path("/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/phase23_20260511_standard_d26_stagec_vs_d3/d3_global_step_300/predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/scratch/erichyu/stageD_d3_hard_ook/diagnostics/d3_ook_uptake_20260511"))
    parser.add_argument("--report-path", type=Path, default=Path("/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/stageD_d3_hard_ook_uptake_diagnosis_2026-05-11.md"))
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "dataset": summarize_dataset(args.train_parquet, args.val_parquet, args.train_batch_size, args.rollout_n, args.steps),
        "training_log": parse_training_steps(args.training_log, args.train_batch_size),
        "phase23": summarize_predictions(args.d3_predictions),
        "comparison": read_comparison(args.comparison_csv),
        "training_log_path": str(args.training_log),
        "comparison_csv": str(args.comparison_csv),
        "d3_predictions": str(args.d3_predictions),
    }
    json_path = args.output_dir / "d3_ook_uptake_diagnosis.json"
    payload["json_path"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.report_path, payload)

    print(f"[PASS] wrote JSON: {json_path}")
    print(f"[PASS] wrote report: {args.report_path}")
    print(
        "[SUMMARY] OOK rows "
        f"{payload['dataset']['train_ook_rows']} / {payload['dataset']['train_rows']} "
        f"({payload['dataset']['train_ook_rate']:.4%}); "
        f"D3 D2.6 abstain {payload['phase23']['d26_hard_ook']['abstain_count']} / {payload['phase23']['d26_hard_ook']['n']}"
    )


if __name__ == "__main__":
    main()
