#!/usr/bin/env python3
"""Build MedQA SFT parquet for cold-start entry (Stage B4)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("pyarrow is required. Please install pyarrow in the current environment.") from exc

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from templates import MEDICAL_SFT_PROMPT_TEMPLATE, TEMPLATE_VERSION
else:
    from .templates import MEDICAL_SFT_PROMPT_TEMPLATE, TEMPLATE_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CSV = REPO_ROOT / "data" / "medical" / "processed_csv" / "medqa_train_sft_half.csv"
DEFAULT_DISTILL_DIR = REPO_ROOT / "data" / "medical" / "distill"
DEFAULT_VERL_DIR = REPO_ROOT / "data" / "medical" / "verl"
DEFAULT_REPORTS_DIR = REPO_ROOT / "data" / "medical" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build medqa_sft_train/val parquet for Stage B4.")
    parser.add_argument("--sft_csv", type=Path, default=DEFAULT_SFT_CSV)
    parser.add_argument("--distill_dir", type=Path, default=DEFAULT_DISTILL_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_VERL_DIR)
    parser.add_argument("--reports_dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--require_distill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, any missing distilled completion will fail the run.",
    )
    parser.add_argument(
        "--placeholder_mode",
        type=str,
        choices=["simple"],
        default="simple",
        help="Used only when --no-require_distill is set.",
    )
    parser.add_argument(
        "--missing_report_path",
        type=Path,
        default=DEFAULT_REPORTS_DIR / "missing_distill_completions.txt",
    )
    return parser.parse_args()


def _parse_json_list(text: str, field_name: str) -> list[str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON field `{field_name}`: {text}") from exc
    if not isinstance(value, list):
        raise ValueError(f"Field `{field_name}` must be a JSON list.")
    return [str(x) for x in value]


def _load_sft_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SFT source CSV: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {"question_id", "question_index", "problem", "target"}
        if not required_fields.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required fields: {required_fields}")

        for row in reader:
            target = _parse_json_list(row["target"], "target")
            rows.append(
                {
                    "question_id": str(row["question_id"]),
                    "question_index": int(row["question_index"]),
                    "problem": str(row["problem"]),
                    "target": target,
                }
            )
    return rows


def _read_completion(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
    return text


def attach_distilled_completion(example: dict[str, Any], distill_dir: Path) -> str | None:
    candidates = [distill_dir / f"{example['question_id']}.txt", distill_dir / f"{example['question_index']}.txt"]
    for candidate in candidates:
        if candidate.exists():
            completion = _read_completion(candidate)
            if completion:
                return completion
    return None


def _placeholder_completion(example: dict[str, Any], mode: str) -> str:
    if mode != "simple":
        raise ValueError(f"Unsupported placeholder mode: {mode}")
    target = example["target"][0] if example["target"] else "A"
    return (
        "<think>\n"
        "I will provide concise medical reasoning based on the question and options.\n"
        "</think>\n"
        "<answer>\n"
        f"The best answer is \\boxed{{{target}}}.\n"
        "</answer>"
    )


def to_sft_record(example: dict[str, Any], completion: str) -> dict[str, str]:
    prompt = MEDICAL_SFT_PROMPT_TEMPLATE.format(question=example["problem"])
    return {"prompt": prompt, "response": completion.strip()}


def train_val_split_for_sft(records: list[dict[str, Any]], ratio: float = 0.9, seed: int = 2026) -> tuple[list, list]:
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"train/val ratio must be in (0, 1), got {ratio}")

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    cut = int(len(indices) * ratio)
    train_ids = set(indices[:cut])
    train_records = [records[i] for i in range(len(records)) if i in train_ids]
    val_records = [records[i] for i in range(len(records)) if i not in train_ids]
    return train_records, val_records


def _write_parquet(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path, compression="snappy")


def _write_missing_report(missing_ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in missing_ids:
            f.write(f"{item}\n")


def main() -> None:
    args = parse_args()
    source_rows = _load_sft_csv(args.sft_csv)

    missing_ids: list[str] = []
    sft_rows: list[dict[str, str]] = []
    distill_dir_exists = args.distill_dir.exists()

    for row in source_rows:
        completion = attach_distilled_completion(row, args.distill_dir) if distill_dir_exists else None
        if completion is None:
            missing_ids.append(row["question_id"])
            if args.require_distill:
                continue
            completion = _placeholder_completion(row, args.placeholder_mode)
        sft_rows.append(to_sft_record(row, completion))

    if args.require_distill and missing_ids:
        _write_missing_report(missing_ids, args.missing_report_path)
        raise SystemExit(
            "Missing distilled completions detected. "
            f"Total missing={len(missing_ids)}. "
            f"Report written to {args.missing_report_path}"
        )

    if not sft_rows:
        raise RuntimeError("No SFT rows generated.")

    train_rows, val_rows = train_val_split_for_sft(sft_rows, ratio=args.train_ratio, seed=args.seed)
    train_out = args.output_dir / "medqa_sft_train.parquet"
    val_out = args.output_dir / "medqa_sft_val.parquet"
    _write_parquet(train_rows, train_out)
    _write_parquet(val_rows, val_out)

    stats = {
        "template_version": TEMPLATE_VERSION,
        "source_csv": str(args.sft_csv),
        "distill_dir": str(args.distill_dir),
        "require_distill": args.require_distill,
        "placeholder_mode": args.placeholder_mode if not args.require_distill else None,
        "split_seed": args.seed,
        "train_ratio": args.train_ratio,
        "counts": {"source_rows": len(source_rows), "generated_rows": len(sft_rows), "train_rows": len(train_rows), "val_rows": len(val_rows)},
        "missing_distill_count": len(missing_ids),
        "missing_report_path": str(args.missing_report_path) if missing_ids else None,
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.reports_dir / "medqa_sft_stageB_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if missing_ids and not args.require_distill:
        _write_missing_report(missing_ids, args.missing_report_path)

    print("[PASS] MedQA SFT Stage-B assets generated.")
    print(f"  medqa_sft_train.parquet: {train_out}")
    print(f"  medqa_sft_val.parquet:   {val_out}")
    print(f"  stats:                   {stats_path}")
    if missing_ids:
        print(f"  missing_distill_count={len(missing_ids)} report={args.missing_report_path}")


if __name__ == "__main__":
    main()

