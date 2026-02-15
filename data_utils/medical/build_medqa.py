#!/usr/bin/env python3
"""Build MedQA Stage-B assets (B1-B3): parse, split, and RL parquet."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("pyarrow is required. Please install pyarrow in the current environment.") from exc

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from templates import MEDICAL_RL_PROMPT_TEMPLATE, TEMPLATE_VERSION
else:
    from .templates import MEDICAL_RL_PROMPT_TEMPLATE, TEMPLATE_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "medical" / "raw"
DEFAULT_PROCESSED_CSV_DIR = REPO_ROOT / "data" / "medical" / "processed_csv"
DEFAULT_VERL_DIR = REPO_ROOT / "data" / "medical" / "verl"
DEFAULT_REPORTS_DIR = REPO_ROOT / "data" / "medical" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MedQA Stage-B data assets.")
    parser.add_argument("--raw_data_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed_csv_dir", type=Path, default=DEFAULT_PROCESSED_CSV_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_VERL_DIR)
    parser.add_argument("--reports_dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--policy", type=str, default="half", choices=["half"])
    parser.add_argument(
        "--grpo_eval_split",
        type=str,
        default="test",
        choices=["dev", "test", "dev+test"],
        help="Which MedQA split(s) will be used as medqa_grpo_test.parquet.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {i + 1}") from exc
    return rows


def load_medqa_split(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def normalize_choices(record: dict[str, Any]) -> list[str]:
    raw_options = record.get("options")
    if not isinstance(raw_options, dict):
        raise ValueError("MedQA record must include dict field `options`.")

    choices: list[str] = []
    for key in sorted(raw_options):
        label = str(key).strip().upper()
        text = str(raw_options[key]).strip()
        if not label or not text:
            raise ValueError("Found empty label/text in options.")
        choices.append(f"{label}: {text}")

    if not choices:
        raise ValueError("Record has no valid choices.")
    return choices


def canonical_answer(record: dict[str, Any]) -> list[str]:
    raw_options = record.get("options")
    if not isinstance(raw_options, dict):
        raise ValueError("MedQA record must include dict field `options`.")

    answer_idx = str(record.get("answer_idx", "")).strip().upper()
    if answer_idx and answer_idx in {str(k).strip().upper() for k in raw_options.keys()}:
        return [answer_idx]

    answer_text = str(record.get("answer", "")).strip()
    if not answer_text:
        raise ValueError("No answer_idx and no answer text found.")

    matched_labels = [
        str(label).strip().upper() for label, option in raw_options.items() if str(option).strip() == answer_text
    ]
    if len(matched_labels) != 1:
        raise ValueError("Cannot map MedQA answer text to a unique choice label.")
    return matched_labels


def render_question_with_choices(question: str, choices: list[str], choice_type: str) -> str:
    lines: list[str] = [question.strip()]
    if choice_type == "multi":
        lines.append("This is a multi-choice question and there may be multiple correct options.")
    lines.extend(choices)
    return "\n".join(lines).strip()


def normalize_medqa_record(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("MedQA record has empty question.")

    choices = normalize_choices(record)
    target = canonical_answer(record)
    choice_type = str(record.get("choice_type", "single")).strip().lower() or "single"
    if choice_type not in {"single", "multi"}:
        choice_type = "single"

    problem = render_question_with_choices(question, choices, choice_type)
    question_id = str(record.get("id") or f"medqa_{split}_{index}")

    return {
        "question_id": question_id,
        "question_index": index,
        "question": question,
        "choice_type": choice_type,
        "choices": choices,
        "target": target,
        "problem": problem,
        "answer_text": str(record.get("answer", "")).strip(),
        "source_split": split,
    }


def split_for_sft_grpo(records: list[dict[str, Any]], policy: str = "half", seed: int = 2026) -> tuple[list, list]:
    if policy != "half":
        raise ValueError(f"Unsupported split policy: {policy}")

    order = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(order)

    cut = len(order) // 2
    sft_ids = set(order[:cut])
    sft_half = [records[i] for i in range(len(records)) if i in sft_ids]
    grpo_half = [records[i] for i in range(len(records)) if i not in sft_ids]
    return sft_half, grpo_half


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return int(sorted_vals[idx])


def _length_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    lengths = [len(item["question"].split()) for item in records]
    if not lengths:
        return {"p50": 0, "p95": 0, "max": 0}
    return {"p50": _percentile(lengths, 0.50), "p95": _percentile(lengths, 0.95), "max": max(lengths)}


def _answer_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for item in records:
        if item["target"]:
            counter[item["target"][0]] += 1
    return dict(sorted(counter.items()))


def ensure_sample_quality(records: list[dict[str, Any]], n: int = 20) -> None:
    for item in records[:n]:
        if not item["choices"]:
            raise ValueError(f"Empty choices in sample question_id={item['question_id']}")
        labels = {choice.split(":", 1)[0].strip() for choice in item["choices"]}
        if not item["target"] or item["target"][0] not in labels:
            raise ValueError(f"Invalid answer mapping in question_id={item['question_id']}")


def to_rl_record(example: dict[str, Any], data_source: str, split: str, index: int) -> dict[str, Any]:
    prompt_text = MEDICAL_RL_PROMPT_TEMPLATE.format(question=example["problem"])
    ground_truth = {
        "schema_version": "medical_v1",
        "target": example["target"],
        "problem": example["problem"],
        "choices": example["choices"],
        "choice_type": example["choice_type"],
        "out_of_knowledge": False,
    }

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "medical",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": index,
            "question_id": example["question_id"],
            "source_dataset": data_source,
        },
    }


def _write_parquet(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path, compression="snappy")


def _write_normalized_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "question_index",
        "question",
        "problem",
        "choice_type",
        "target",
        "choices",
        "answer_text",
        "source_split",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "question_index": row["question_index"],
                    "question": row["question"],
                    "problem": row["problem"],
                    "choice_type": row["choice_type"],
                    "target": json.dumps(row["target"], ensure_ascii=False),
                    "choices": json.dumps(row["choices"], ensure_ascii=False),
                    "answer_text": row["answer_text"],
                    "source_split": row["source_split"],
                }
            )


def _collect_stats(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    sft_half: list[dict[str, Any]],
    grpo_half: list[dict[str, Any]],
    grpo_eval_records: list[dict[str, Any]],
    seed: int,
    policy: str,
    grpo_eval_split: str,
) -> dict[str, Any]:
    return {
        "template_version": TEMPLATE_VERSION,
        "split_seed": seed,
        "split_policy": policy,
        "grpo_eval_split": grpo_eval_split,
        "counts": {
            "medqa_train": len(train_records),
            "medqa_dev": len(dev_records),
            "medqa_test": len(test_records),
            "train_sft_half": len(sft_half),
            "train_grpo_half": len(grpo_half),
            "grpo_eval_records": len(grpo_eval_records),
        },
        "answer_distribution": {
            "medqa_train": _answer_distribution(train_records),
            "train_sft_half": _answer_distribution(sft_half),
            "train_grpo_half": _answer_distribution(grpo_half),
        },
        "question_length": {
            "medqa_train": _length_stats(train_records),
            "train_sft_half": _length_stats(sft_half),
            "train_grpo_half": _length_stats(grpo_half),
        },
    }


def main() -> None:
    args = parse_args()

    medqa_train = [normalize_medqa_record(r, "train", i) for i, r in enumerate(load_medqa_split(args.raw_data_dir / "medqa_train.jsonl"))]
    medqa_dev = [normalize_medqa_record(r, "dev", i) for i, r in enumerate(load_medqa_split(args.raw_data_dir / "medqa_dev.jsonl"))]
    medqa_test = [normalize_medqa_record(r, "test", i) for i, r in enumerate(load_medqa_split(args.raw_data_dir / "medqa_test.jsonl"))]

    ensure_sample_quality(medqa_train, n=20)
    ensure_sample_quality(medqa_dev, n=20)
    ensure_sample_quality(medqa_test, n=20)

    sft_half, grpo_half = split_for_sft_grpo(medqa_train, policy=args.policy, seed=args.seed)
    sft_half_2, grpo_half_2 = split_for_sft_grpo(medqa_train, policy=args.policy, seed=args.seed)
    if [x["question_id"] for x in sft_half] != [x["question_id"] for x in sft_half_2]:
        raise RuntimeError("Split reproducibility check failed for sft_half.")
    if [x["question_id"] for x in grpo_half] != [x["question_id"] for x in grpo_half_2]:
        raise RuntimeError("Split reproducibility check failed for grpo_half.")

    sft_ids = {item["question_id"] for item in sft_half}
    grpo_ids = {item["question_id"] for item in grpo_half}
    overlap = sft_ids.intersection(grpo_ids)
    if overlap:
        raise RuntimeError(f"Found overlap between sft_half and grpo_half, e.g. {next(iter(overlap))}")

    if args.grpo_eval_split == "dev":
        grpo_eval_records = medqa_dev
    elif args.grpo_eval_split == "test":
        grpo_eval_records = medqa_test
    else:
        grpo_eval_records = medqa_dev + medqa_test

    grpo_train_parquet = args.output_dir / "medqa_grpo_train.parquet"
    grpo_test_parquet = args.output_dir / "medqa_grpo_test.parquet"

    rl_train_rows = [to_rl_record(item, "medqa", "train_grpo_half", idx) for idx, item in enumerate(grpo_half)]
    rl_eval_rows = [to_rl_record(item, "medqa", item["source_split"], idx) for idx, item in enumerate(grpo_eval_records)]
    _write_parquet(rl_train_rows, grpo_train_parquet)
    _write_parquet(rl_eval_rows, grpo_test_parquet)

    _write_normalized_csv(medqa_train, args.processed_csv_dir / "medqa_train.csv")
    _write_normalized_csv(medqa_dev, args.processed_csv_dir / "medqa_dev.csv")
    _write_normalized_csv(medqa_test, args.processed_csv_dir / "medqa_test.csv")
    _write_normalized_csv(sft_half, args.processed_csv_dir / "medqa_train_sft_half.csv")
    _write_normalized_csv(grpo_half, args.processed_csv_dir / "medqa_train_grpo_half.csv")

    stats = _collect_stats(
        train_records=medqa_train,
        dev_records=medqa_dev,
        test_records=medqa_test,
        sft_half=sft_half,
        grpo_half=grpo_half,
        grpo_eval_records=grpo_eval_records,
        seed=args.seed,
        policy=args.policy,
        grpo_eval_split=args.grpo_eval_split,
    )
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.reports_dir / "medqa_stageB_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[PASS] MedQA Stage-B assets generated.")
    print(f"  medqa_grpo_train.parquet: {grpo_train_parquet}")
    print(f"  medqa_grpo_test.parquet:  {grpo_test_parquet}")
    print(f"  medqa_train_sft_half.csv: {args.processed_csv_dir / 'medqa_train_sft_half.csv'}")
    print(f"  stats:                    {stats_path}")
    print(f"  counts: train={len(medqa_train)}, dev={len(medqa_dev)}, test={len(medqa_test)}, sft_half={len(sft_half)}, grpo_half={len(grpo_half)}")


if __name__ == "__main__":
    main()

