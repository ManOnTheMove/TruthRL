#!/usr/bin/env python3
"""Stage D post-training decoded eval harness.

This script evaluates one HF-style model on:
- standard MedQA validation/test parquet
- D2.6 hard-OOK rows from the OOK-augmented train parquet

It writes per-example predictions and per-group metrics. It does not train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer



def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run Stage D decoded eval for one model.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--standard-parquet",
        type=Path,
        default=repo_root / "data" / "medical" / "verl" / "medqa_grpo_test.parquet",
    )
    parser.add_argument(
        "--d26-parquet",
        type=Path,
        default=repo_root / "data" / "medical" / "verl" / "medqa_grpo_train_with_ook_d26.parquet",
    )
    parser.add_argument("--groups", default="standard_medqa,d26_hard_ook")
    parser.add_argument("--question-limit-per-group", type=int, default=0)
    parser.add_argument("--include-prompt-text", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--stop-str", default="</answer>")
    parser.add_argument("--include-stop-str-in-output", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_reward_module(repo_root: Path) -> Any:
    reward_path = repo_root / "training" / "verl" / "verl" / "utils" / "reward_score" / "clinical_medqa_reward.py"
    spec = importlib.util.spec_from_file_location("clinical_medqa_reward", str(reward_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward module from {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_set(reward_module: Any, ground_truth: dict[str, Any]) -> set[str]:
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    out: set[str] = set()
    if isinstance(targets, list):
        for item in targets:
            value = reward_module.normalize_choice(str(item))
            if value:
                out.add(value)
    return out


def choice_set(reward_module: Any, ground_truth: dict[str, Any]) -> set[str]:
    choices = ground_truth.get("choices", [])
    out: set[str] = set()
    if isinstance(choices, list):
        for item in choices:
            if not isinstance(item, str):
                continue
            value = reward_module.normalize_choice(item)
            if value:
                out.add(value)
                continue
            if ":" in item:
                value = reward_module.normalize_choice(item.split(":", 1)[0])
                if value:
                    out.add(value)
    return out


def prompt_text(tokenizer: Any, prompt_obj: Any) -> str:
    if isinstance(prompt_obj, list):
        return tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True, tokenize=False)
    return str(prompt_obj)


def load_examples(
    *,
    args: argparse.Namespace,
    reward_module: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    groups = {x.strip() for x in args.groups.split(",") if x.strip()}
    examples: list[dict[str, Any]] = []

    def add_rows(path: Path, source_group: str, only_ook: bool) -> None:
        table = pq.read_table(path, columns=["data_source", "prompt", "reward_model", "extra_info"])
        rows = table.to_pylist()
        added = 0
        for row_idx, row in enumerate(rows):
            reward_model = row.get("reward_model") or {}
            gt = reward_model.get("ground_truth") or {}
            if only_ook and not bool(gt.get("out_of_knowledge", False)):
                continue
            extra_info = row.get("extra_info") or {}
            qid = str(extra_info.get("question_id", f"row_{row_idx}"))
            ptxt = prompt_text(tokenizer, row.get("prompt"))
            targets = sorted(target_set(reward_module, gt))
            choices = sorted(choice_set(reward_module, gt))
            examples.append(
                {
                    "example_id": f"{source_group}:{qid}",
                    "source_group": source_group,
                    "source_parquet": str(path),
                    "row_index": int(row_idx),
                    "question_id": qid,
                    "data_source": str(row.get("data_source", "")),
                    "source_dataset": str(extra_info.get("source_dataset", "")),
                    "split": str(extra_info.get("split", "")),
                    "prompt_text": ptxt,
                    "prompt_sha256": hashlib.sha256(ptxt.encode("utf-8")).hexdigest(),
                    "ground_truth": gt,
                    "target_choices": targets,
                    "choice_set": choices,
                    "is_out_of_knowledge": bool(gt.get("out_of_knowledge", False)),
                    "problem": str(gt.get("problem", "")),
                    "choices": gt.get("choices", []),
                }
            )
            added += 1
            if args.question_limit_per_group > 0 and added >= args.question_limit_per_group:
                break

    if "standard_medqa" in groups:
        add_rows(args.standard_parquet, "standard_medqa", only_ook=False)
    if "d26_hard_ook" in groups:
        add_rows(args.d26_parquet, "d26_hard_ook", only_ook=True)
    if not examples:
        raise RuntimeError("no eval examples selected")
    return examples


def parse_prediction(reward_module: Any, text: str, targets: set[str], choices: set[str]) -> dict[str, Any]:
    parsed = reward_module.parse_medical_answer(
        text,
        policy=reward_module.POLICY_SINGLE_BOXED_STRICT,
        choice_set=choices,
    )
    parsed_choice = parsed.normalized_choice if parsed.parse_status in {"ok_choice", "choice_out_of_set"} else ""
    if parsed.is_abstain:
        parsed_choice = "I don't know"

    return {
        "boxed_raw": parsed.boxed_raw,
        "parsed_choice": parsed_choice,
        "normalized_choice": parsed.normalized_choice,
        "is_abstain": int(parsed.is_abstain),
        "is_boxed_valid": int(parsed.parse_status in {"ok_choice", "ok_abstain"}),
        "is_correct": int(parsed.parse_status == "ok_choice" and parsed.normalized_choice in targets),
        "parse_status": parsed.parse_status,
        "parser_policy": parsed.policy,
        "boxed_count": int(parsed.boxed_count),
        "used_answer_block": int(parsed.used_answer_block),
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def ratio(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def summarize_group(group_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    parse_counts = Counter(str(x["parse_status"]) for x in rows)
    correct = sum(int(x["is_correct"]) for x in rows)
    abstain = sum(int(x["is_abstain"]) for x in rows)
    boxed_valid = sum(int(x["is_boxed_valid"]) for x in rows)
    valid_answers = sum(1 for x in rows if x["parse_status"] == "ok_choice")
    valid_wrong = sum(1 for x in rows if x["parse_status"] == "ok_choice" and not int(x["is_correct"]))
    parse_fail = n - boxed_valid
    rewards = [float(x["d3_score"]) for x in rows if x.get("d3_score") is not None]
    generated_tokens = [float(x["generated_tokens"]) for x in rows if x.get("generated_tokens") is not None]
    return {
        "source_group": group_name,
        "n": n,
        "ook_count": sum(1 for x in rows if bool(x["is_out_of_knowledge"])),
        "correct_count": correct,
        "total_accuracy": ratio(correct, n),
        "valid_answer_count": valid_answers,
        "coverage": ratio(valid_answers, n),
        "selective_accuracy": ratio(correct, valid_answers),
        "abstain_count": abstain,
        "abstain_rate": ratio(abstain, n),
        "boxed_valid_count": boxed_valid,
        "boxed_valid_rate": ratio(boxed_valid, n),
        "parse_fail_count": parse_fail,
        "parse_fail_rate": ratio(parse_fail, n),
        "valid_non_abstain_wrong_count": valid_wrong,
        "valid_non_abstain_wrong_rate": ratio(valid_wrong, n),
        "answered_error_rate": ratio(valid_wrong, valid_answers),
        "d3_reward_mean": mean(rewards),
        "generated_tokens_mean": mean(generated_tokens),
        "parse_status_counts": dict(parse_counts),
    }


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = list(records)
    for row in rows:
        grouped[str(row["source_group"])].append(row)

    summaries: list[dict[str, Any]] = []
    if rows:
        summaries.append(summarize_group("all", rows))
    for group_name in sorted(grouped):
        summaries.append(summarize_group(group_name, grouped[group_name]))
    return summaries


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_id",
        "source_group",
        "n",
        "ook_count",
        "correct_count",
        "total_accuracy",
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
        "valid_non_abstain_wrong_rate",
        "answered_error_rate",
        "d3_reward_mean",
        "generated_tokens_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    started_at = utc_now()

    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"model path invalid: {args.model_path}")
    out_dir = args.output_root / args.run_id / args.model_id
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output dir exists: {out_dir}; pass --overwrite to replace files")
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_module = load_reward_module(repo_root)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    examples = load_examples(args=args, reward_module=reward_module, tokenizer=tokenizer)
    prompts = [x["prompt_text"] for x in examples]

    manifest = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "model_path": str(args.model_path),
        "started_at_utc": started_at,
        "output_dir": str(out_dir),
        "example_count": len(examples),
        "groups": args.groups,
        "parser": {
            "policy": reward_module.POLICY_SINGLE_BOXED_STRICT,
            "module": "verl.utils.reward_score.medical_answer_parser",
        },
        "standard_parquet": str(args.standard_parquet),
        "d26_parquet": str(args.d26_parquet),
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
            "stop_str": args.stop_str,
            "include_stop_str_in_output": args.include_stop_str_in_output,
            "seed": args.seed,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        max_num_batched_tokens=int(args.max_num_batched_tokens),
    )

    sampling_kwargs: dict[str, Any] = {
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_new_tokens),
        "stop": [args.stop_str] if args.stop_str else None,
        "include_stop_str_in_output": bool(args.include_stop_str_in_output),
        "skip_special_tokens": True,
        "seed": int(args.seed),
    }
    if args.temperature > 0:
        sampling_kwargs.update({"top_p": float(args.top_p), "top_k": int(args.top_k), "min_p": float(args.min_p)})
    sampling_params = SamplingParams(**sampling_kwargs)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    if len(outputs) != len(examples):
        raise RuntimeError(f"expected {len(examples)} outputs, got {len(outputs)}")

    records: list[dict[str, Any]] = []
    for example, output in zip(examples, outputs):
        if not output.outputs:
            response_text = ""
            finish_reason = "missing_output"
            generated_tokens = 0
        else:
            first = output.outputs[0]
            response_text = first.text if isinstance(first.text, str) else str(first.text)
            finish_reason = str(getattr(first, "finish_reason", "unknown"))
            token_ids = getattr(first, "token_ids", None)
            generated_tokens = int(len(token_ids)) if token_ids is not None else 0

        targets = set(example["target_choices"])
        choices = set(example["choice_set"])
        parsed = parse_prediction(reward_module, response_text, targets, choices)
        score_info = reward_module.compute_score(
            data_source="medical",
            solution_str=response_text,
            ground_truth=example["ground_truth"],
            stage_mode="d3_hard_ook",
            k=1.0,
            enable_format=False,
            enable_consistency=False,
        )
        record = {
            "run_id": args.run_id,
            "model_id": args.model_id,
            "model_path": str(args.model_path),
            "example_id": example["example_id"],
            "source_group": example["source_group"],
            "source_parquet": example["source_parquet"],
            "row_index": example["row_index"],
            "question_id": example["question_id"],
            "data_source": example["data_source"],
            "source_dataset": example["source_dataset"],
            "split": example["split"],
            "prompt_sha256": example["prompt_sha256"],
            "target_choices": example["target_choices"],
            "choice_set": example["choice_set"],
            "is_out_of_knowledge": bool(example["is_out_of_knowledge"]),
            "problem": example["problem"],
            "choices_json": json.dumps(example["choices"], ensure_ascii=False),
            "response_text": response_text,
            "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            "finish_reason": finish_reason,
            "generated_tokens": generated_tokens,
            "created_at_utc": utc_now(),
            **parsed,
            "d3_score": float(score_info["score"]),
            "d3_outcome_score": float(score_info["outcome_score"]),
            "d3_prediction_type": int(score_info["prediction_type"]),
        }
        if args.include_prompt_text:
            record["prompt_text"] = example["prompt_text"]
        records.append(record)

    metrics = summarize(records)
    for row in metrics:
        row["model_id"] = args.model_id

    predictions_jsonl = out_dir / "predictions.jsonl"
    predictions_parquet = out_dir / "predictions.parquet"
    metrics_json = out_dir / "metrics_by_group.json"
    metrics_csv = out_dir / "metrics_by_group.csv"

    write_jsonl(predictions_jsonl, records)
    pq.write_table(pa.Table.from_pylist(records), predictions_parquet, compression="zstd")
    metrics_json.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "parser": {
                    "policy": reward_module.POLICY_SINGLE_BOXED_STRICT,
                    "module": "verl.utils.reward_score.medical_answer_parser",
                },
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_metrics_csv(metrics_csv, metrics)

    manifest["ended_at_utc"] = utc_now()
    manifest["outputs"] = {
        "predictions_jsonl": str(predictions_jsonl),
        "predictions_parquet": str(predictions_parquet),
        "metrics_json": str(metrics_json),
        "metrics_csv": str(metrics_csv),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[PASS] model_id={args.model_id} examples={len(records)}")
    print(f"[INFO] predictions={predictions_parquet}")
    print(f"[INFO] metrics={metrics_csv}")


if __name__ == "__main__":
    main()
