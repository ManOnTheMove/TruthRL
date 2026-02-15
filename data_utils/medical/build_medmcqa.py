#!/usr/bin/env python3
"""Build MedMCQA Stage-B assets: parser rewrite + evaluation parquet."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


ANSWER_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MedMCQA evaluation parquet for Stage-B.")
    parser.add_argument("--raw_data_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed_csv_dir", type=Path, default=DEFAULT_PROCESSED_CSV_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_VERL_DIR)
    parser.add_argument("--reports_dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--splits", type=str, default="dev,test", help="Comma-separated splits, e.g. dev,test")
    parser.add_argument(
        "--drop_missing_target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop samples without answer labels when building eval parquet.",
    )
    return parser.parse_args()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.startswith("version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{path} is a git-lfs pointer, not the actual dataset.")
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {i + 1}") from exc
    return rows


def load_medmcqa_split(path: Path) -> list[dict[str, Any]]:
    return _read_json_lines(path)


def normalize_choices(record: dict[str, Any]) -> list[str]:
    options = {
        "A": str(record.get("opa", "")).strip(),
        "B": str(record.get("opb", "")).strip(),
        "C": str(record.get("opc", "")).strip(),
        "D": str(record.get("opd", "")).strip(),
    }
    choices = [f"{k}: {v}" for k, v in options.items() if v]
    if not choices:
        raise ValueError("MedMCQA record has no valid options.")
    return choices


def canonical_answer(record: dict[str, Any]) -> list[str]:
    cop = record.get("cop")
    if cop is None or cop == "":
        return []

    try:
        numeric = int(cop)
    except (TypeError, ValueError):
        return []

    mapped = ANSWER_MAP.get(numeric)
    if mapped is None:
        return []
    return [mapped]


def render_question_with_choices(question: str, choices: list[str], choice_type: str) -> str:
    lines = [question.strip()]
    if choice_type == "multi":
        lines.append("This is a multi-choice question and there may be multiple correct options.")
    lines.extend(choices)
    return "\n".join(lines).strip()


def normalize_medmcqa_record(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("MedMCQA record has empty question.")

    choices = normalize_choices(record)
    choice_type = str(record.get("choice_type", "single")).strip().lower() or "single"
    if choice_type not in {"single", "multi"}:
        choice_type = "single"
    target = canonical_answer(record)
    problem = render_question_with_choices(question, choices, choice_type)
    question_id = str(record.get("id") or f"medmcqa_{split}_{index}")

    return {
        "question_id": question_id,
        "question_index": index,
        "question": question,
        "choice_type": choice_type,
        "choices": choices,
        "target": target,
        "problem": problem,
        "source_split": split,
    }


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
        if item["target"] and item["target"][0] not in labels:
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
    fields = ["question_id", "question_index", "question", "problem", "choice_type", "target", "choices", "source_split"]
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
                    "source_split": row["source_split"],
                }
            )


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No split specified.")

    normalized_by_split: dict[str, list[dict[str, Any]]] = {}
    dropped_missing_target = 0

    for split in splits:
        path = args.raw_data_dir / f"medmcqa_{split}.json"
        raw_records = load_medmcqa_split(path)
        normalized = [normalize_medmcqa_record(r, split, i) for i, r in enumerate(raw_records)]
        ensure_sample_quality(normalized, n=20)
        normalized_by_split[split] = normalized
        _write_normalized_csv(normalized, args.processed_csv_dir / f"medmcqa_{split}.csv")

    eval_source_records: list[dict[str, Any]] = []
    for split in splits:
        for row in normalized_by_split[split]:
            if row["target"]:
                eval_source_records.append(row)
            elif args.drop_missing_target:
                dropped_missing_target += 1
            else:
                eval_source_records.append(row)

    if not eval_source_records:
        raise RuntimeError("No records available for medmcqa_eval.parquet after filtering.")

    eval_rows = [to_rl_record(item, "medmcqa", item["source_split"], idx) for idx, item in enumerate(eval_source_records)]
    eval_out = args.output_dir / "medmcqa_eval.parquet"
    _write_parquet(eval_rows, eval_out)

    stats = {
        "template_version": TEMPLATE_VERSION,
        "splits": splits,
        "counts": {split: len(normalized_by_split[split]) for split in splits},
        "eval_count": len(eval_source_records),
        "dropped_missing_target": dropped_missing_target,
        "answer_distribution": _answer_distribution(eval_source_records),
        "question_length": _length_stats(eval_source_records),
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.reports_dir / "medmcqa_stageB_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[PASS] MedMCQA Stage-B assets generated.")
    print(f"  medmcqa_eval.parquet: {eval_out}")
    print(f"  stats:                {stats_path}")
    print(f"  eval_count={len(eval_source_records)} dropped_missing_target={dropped_missing_target}")


if __name__ == "__main__":
    main()

