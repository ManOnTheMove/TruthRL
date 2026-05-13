#!/usr/bin/env python3
"""Compare Stage D eval metrics across models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ORDERED_METRICS = [
    "n",
    "total_accuracy",
    "coverage",
    "selective_accuracy",
    "abstain_rate",
    "boxed_valid_rate",
    "parse_fail_rate",
    "valid_non_abstain_wrong_rate",
    "answered_error_rate",
    "d3_reward_mean",
    "generated_tokens_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage D eval summaries.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-ids", nargs="+", required=True)
    parser.add_argument("--baseline-model-id", default="stagec_base")
    parser.add_argument("--candidate-model-id", default="d3_global_step_300")
    return parser.parse_args()


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def load_metrics(run_dir: Path, model_ids: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for model_id in model_ids:
        path = run_dir / model_id / "metrics_by_group.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["metrics"]:
            out[(model_id, str(row["source_group"]))] = row
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["source_group", "metric"] + [k for k in rows[0].keys() if k not in {"source_group", "metric"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        line = "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        lines.append(line)
        if idx == 0:
            lines.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return lines


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    metrics = load_metrics(run_dir, args.model_ids)
    groups = sorted({group for _, group in metrics.keys()})

    comparison_rows: list[dict[str, Any]] = []
    for group in groups:
        for metric in ORDERED_METRICS:
            row: dict[str, Any] = {"source_group": group, "metric": metric}
            for model_id in args.model_ids:
                row[model_id] = metrics.get((model_id, group), {}).get(metric)
            base = row.get(args.baseline_model_id)
            cand = row.get(args.candidate_model_id)
            if isinstance(base, (int, float)) and isinstance(cand, (int, float)):
                row[f"{args.candidate_model_id}_minus_{args.baseline_model_id}"] = float(cand - base)
            else:
                row[f"{args.candidate_model_id}_minus_{args.baseline_model_id}"] = None
            comparison_rows.append(row)

    compare_csv = run_dir / "comparison_by_group.csv"
    write_csv(compare_csv, comparison_rows)

    md_lines = [
        "# Stage D Phase 2-3 Eval Comparison",
        "",
        f"- run_id: `{args.run_id}`",
        f"- output_root: `{args.output_root}`",
        f"- baseline_model_id: `{args.baseline_model_id}`",
        f"- candidate_model_id: `{args.candidate_model_id}`",
        f"- comparison_csv: `{compare_csv}`",
        "",
        "## Interpretation Guardrail",
        "",
        "This report compares decoded eval metrics for Stage C base and D3 global_step_300. It supports post-training behavior analysis only. It does not by itself prove held-out OOK generalization, because the D2.6 hard-OOK set is train-side.",
        "",
    ]

    for group in groups:
        md_lines.extend([f"## Group: `{group}`", ""])
        table_rows = [["metric", *args.model_ids, "delta"]]
        for metric in ORDERED_METRICS:
            row = next(x for x in comparison_rows if x["source_group"] == group and x["metric"] == metric)
            table_rows.append(
                [
                    metric,
                    *[fmt(row.get(model_id)) for model_id in args.model_ids],
                    fmt(row.get(f"{args.candidate_model_id}_minus_{args.baseline_model_id}")),
                ]
            )
        md_lines.extend(markdown_table(table_rows))
        md_lines.append("")

    report_path = run_dir / "phase2_phase3_eval_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[PASS] wrote {compare_csv}")
    print(f"[PASS] wrote {report_path}")


if __name__ == "__main__":
    main()
