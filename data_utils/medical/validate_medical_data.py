#!/usr/bin/env python3
"""Validate Stage-B medical data assets and provide OOK update interface."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("pyarrow is required. Please install pyarrow in the current environment.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERL_DIR = REPO_ROOT / "data" / "medical" / "verl"
DEFAULT_REPORTS_DIR = REPO_ROOT / "data" / "medical" / "reports"

RL_REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
SFT_REQUIRED_COLUMNS = {"prompt", "response"}
GROUND_TRUTH_REQUIRED_KEYS = {"schema_version", "target", "problem", "choices", "choice_type", "out_of_knowledge"}
EXTRA_INFO_REQUIRED_KEYS = {"split", "index", "question_id", "source_dataset"}


class ValidationError(Exception):
    """Raised when schema/data validation fails."""


@dataclass
class FileValidationSummary:
    path: str
    row_count: int
    columns: list[str]
    unique_question_ids: int | None = None
    duplicate_question_ids: int | None = None
    prompt_boxed_contract_violations: int | None = None
    empty_target_count: int | None = None
    out_of_knowledge_true_count: int | None = None
    answer_distribution: dict[str, int] | None = None


def _read_parquet(path: Path) -> pa.Table:
    if not path.exists():
        raise ValidationError(f"Missing parquet file: {path}")
    try:
        return pq.read_table(path)
    except Exception as exc:
        raise ValidationError(f"Failed to read parquet: {path}") from exc


def validate_prompt_contract(prompt: Any) -> bool:
    """RL prompt contract: must include boxed-answer instruction."""
    if not isinstance(prompt, list) or not prompt:
        return False
    first = prompt[0]
    if not isinstance(first, dict):
        return False
    content = first.get("content")
    if not isinstance(content, str):
        return False
    return "\\boxed{" in content


def validate_rl_schema(path: Path) -> FileValidationSummary:
    table = _read_parquet(path)
    columns = set(table.column_names)
    missing = RL_REQUIRED_COLUMNS - columns
    if missing:
        raise ValidationError(f"{path}: missing RL columns {sorted(missing)}")

    rows = table.to_pylist()
    qid_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    prompt_violations = 0
    empty_target_count = 0
    ook_true_count = 0

    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        if not validate_prompt_contract(prompt):
            prompt_violations += 1

        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict):
            raise ValidationError(f"{path}: row {i} has invalid reward_model type")
        ground_truth = reward_model.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise ValidationError(f"{path}: row {i} has invalid ground_truth type")

        gt_missing = GROUND_TRUTH_REQUIRED_KEYS - set(ground_truth.keys())
        if gt_missing:
            raise ValidationError(f"{path}: row {i} missing ground_truth keys {sorted(gt_missing)}")

        target = ground_truth.get("target")
        if not isinstance(target, list) or len(target) == 0:
            empty_target_count += 1
        else:
            answer_counter[str(target[0])] += 1

        out_of_knowledge = ground_truth.get("out_of_knowledge")
        if not isinstance(out_of_knowledge, bool):
            raise ValidationError(f"{path}: row {i} out_of_knowledge must be bool")
        if out_of_knowledge:
            ook_true_count += 1

        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValidationError(f"{path}: row {i} has invalid extra_info type")
        extra_missing = EXTRA_INFO_REQUIRED_KEYS - set(extra_info.keys())
        if extra_missing:
            raise ValidationError(f"{path}: row {i} missing extra_info keys {sorted(extra_missing)}")

        qid = extra_info.get("question_id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValidationError(f"{path}: row {i} has invalid question_id")
        qid_counter[qid] += 1

    duplicate_count = sum(1 for _, cnt in qid_counter.items() if cnt > 1)
    return FileValidationSummary(
        path=str(path),
        row_count=table.num_rows,
        columns=list(table.column_names),
        unique_question_ids=len(qid_counter),
        duplicate_question_ids=duplicate_count,
        prompt_boxed_contract_violations=prompt_violations,
        empty_target_count=empty_target_count,
        out_of_knowledge_true_count=ook_true_count,
        answer_distribution=dict(sorted(answer_counter.items())),
    )


def validate_sft_schema(path: Path) -> FileValidationSummary:
    table = _read_parquet(path)
    columns = set(table.column_names)
    missing = SFT_REQUIRED_COLUMNS - columns
    if missing:
        raise ValidationError(f"{path}: missing SFT columns {sorted(missing)}")

    rows = table.to_pylist()
    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError(f"{path}: row {i} invalid prompt")
        if not isinstance(response, str) or not response.strip():
            raise ValidationError(f"{path}: row {i} invalid response")

    return FileValidationSummary(
        path=str(path),
        row_count=table.num_rows,
        columns=list(table.column_names),
    )


def _load_ook_mapping(ook_json_path: Path) -> dict[str, bool]:
    if not ook_json_path.exists():
        raise FileNotFoundError(f"OOK label file not found: {ook_json_path}")

    with ook_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, bool] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            mapping[str(key)] = bool(value)
        return mapping

    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid OOK list item at index {i}")
            qid = item.get("question_id")
            if qid is None:
                raise ValueError(f"Missing question_id at OOK list item {i}")
            if "out_of_knowledge" in item:
                mapping[str(qid)] = bool(item["out_of_knowledge"])
            elif "ook" in item:
                mapping[str(qid)] = bool(item["ook"])
            else:
                raise ValueError(f"Missing out_of_knowledge/ook field in OOK list item {i}")
        return mapping

    raise ValueError("OOK json must be a dict or list")


def update_ook_labels(input_parquet: str | Path, ook_json: str | Path, output_parquet: str | Path) -> dict[str, Any]:
    input_path = Path(input_parquet)
    ook_path = Path(ook_json)
    output_path = Path(output_parquet)

    table = _read_parquet(input_path)
    rows = table.to_pylist()
    mapping = _load_ook_mapping(ook_path)

    updated = 0
    missing_in_parquet = 0
    seen_qids: set[str] = set()
    for i, row in enumerate(rows):
        extra_info = row.get("extra_info")
        reward_model = row.get("reward_model")
        if not isinstance(extra_info, dict) or not isinstance(reward_model, dict):
            raise ValidationError(f"{input_path}: row {i} missing extra_info/reward_model")
        qid = extra_info.get("question_id")
        if not isinstance(qid, str):
            raise ValidationError(f"{input_path}: row {i} invalid question_id")
        seen_qids.add(qid)

        ground_truth = reward_model.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise ValidationError(f"{input_path}: row {i} invalid ground_truth")
        if qid in mapping:
            ground_truth["out_of_knowledge"] = bool(mapping[qid])
            updated += 1

    for qid in mapping:
        if qid not in seen_qids:
            missing_in_parquet += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_table = pa.Table.from_pylist(rows)
    pq.write_table(output_table, output_path, compression="snappy")

    summary = {
        "input_parquet": str(input_path),
        "output_parquet": str(output_path),
        "ook_json": str(ook_path),
        "input_rows": len(rows),
        "mapping_size": len(mapping),
        "updated_rows": updated,
        "mapping_question_ids_missing_in_parquet": missing_in_parquet,
    }
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _to_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# Stage B Data Validation Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Hostname: `{report['hostname']}`")
    lines.append(f"- Validation status: `{report['status']}`")
    lines.append("")
    lines.append("## RL Files")
    for item in report["rl_files"]:
        lines.append(f"- `{item['path']}`")
        lines.append(f"  - rows: {item['row_count']}")
        lines.append(f"  - duplicate_question_ids: {item['duplicate_question_ids']}")
        lines.append(f"  - prompt_boxed_contract_violations: {item['prompt_boxed_contract_violations']}")
        lines.append(f"  - empty_target_count: {item['empty_target_count']}")
        lines.append(f"  - out_of_knowledge_true_count: {item['out_of_knowledge_true_count']}")
    lines.append("")
    lines.append("## SFT Files")
    for item in report["sft_files"]:
        lines.append(f"- `{item['path']}`")
        lines.append(f"  - rows: {item['row_count']}")
    lines.append("")
    lines.append("## Errors")
    if report["errors"]:
        for err in report["errors"]:
            lines.append(f"- {err}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def run_validation(
    rl_files: list[Path],
    sft_files: list[Path],
    report_json: Path,
    report_md: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    rl_summaries: list[dict[str, Any]] = []
    sft_summaries: list[dict[str, Any]] = []

    for path in rl_files:
        try:
            summary = validate_rl_schema(path)
            rl_summaries.append(asdict(summary))
        except Exception as exc:
            errors.append(str(exc))

    for path in sft_files:
        try:
            summary = validate_sft_schema(path)
            sft_summaries.append(asdict(summary))
        except Exception as exc:
            errors.append(str(exc))

    report = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "status": "pass" if not errors else "fail",
        "rl_files": rl_summaries,
        "sft_files": sft_summaries,
        "errors": errors,
    }
    _write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_to_markdown(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage-B medical parquet files and update OOK labels.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate RL/SFT parquet schema and generate reports.")
    validate_parser.add_argument(
        "--rl-files",
        nargs="+",
        type=Path,
        default=[
            DEFAULT_VERL_DIR / "medqa_grpo_train.parquet",
            DEFAULT_VERL_DIR / "medqa_grpo_test.parquet",
            DEFAULT_VERL_DIR / "medmcqa_eval.parquet",
        ],
    )
    validate_parser.add_argument(
        "--sft-files",
        nargs="+",
        type=Path,
        default=[DEFAULT_VERL_DIR / "medqa_sft_train.parquet", DEFAULT_VERL_DIR / "medqa_sft_val.parquet"],
    )
    validate_parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORTS_DIR / "stageB_data_report.json")
    validate_parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORTS_DIR / "stageB_data_report.md")

    update_parser = subparsers.add_parser("update-ook", help="Update out_of_knowledge labels in RL parquet.")
    update_parser.add_argument("--input-parquet", type=Path, required=True)
    update_parser.add_argument("--ook-json", type=Path, required=True)
    update_parser.add_argument("--output-parquet", type=Path, required=True)
    update_parser.add_argument("--summary-json", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "validate":
        report = run_validation(args.rl_files, args.sft_files, args.report_json, args.report_md)
        if report["status"] != "pass":
            raise SystemExit(f"Validation failed. See report: {args.report_json}")
        print("[PASS] Stage-B medical data validation passed.")
        print(f"  report_json: {args.report_json}")
        print(f"  report_md:   {args.report_md}")
    elif args.command == "update-ook":
        summary = update_ook_labels(args.input_parquet, args.ook_json, args.output_parquet)
        if args.summary_json is not None:
            _write_json(args.summary_json, summary)
        print("[PASS] OOK labels updated.")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

