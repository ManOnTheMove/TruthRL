#!/usr/bin/env python3
"""Analyze response tag quality in a parquet SFT file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze response tags in parquet.")
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    arr = pq.read_table(args.parquet, columns=["response"]).column("response").to_pylist()
    n = len(arr)
    c: Counter[str] = Counter()
    for r in arr:
        s = (r or "").strip()
        c["has_open_think"] += "<think>" in s
        c["has_close_think"] += "</think>" in s
        c["has_open_answer"] += "<answer>" in s
        c["has_close_answer"] += "</answer>" in s
        c["has_boxed"] += "\\boxed{" in s
        c["start_close_think"] += s.startswith("</think>")
        c["start_close_answer"] += s.startswith("</answer>")
        c["only_close_answer"] += s == "</answer>"
        c["empty"] += s == ""
        c["dup_close_think"] += s.count("</think>") >= 2
        c["dup_close_answer"] += s.count("</answer>") >= 2

    out = {
        "N": n,
        "stats": {k: {"count": int(v), "ratio": (v / n if n else 0.0)} for k, v in c.items()},
        "sample0_head": arr[0][:220] if n else "",
        "sample0_tail": arr[0][-220:] if n else "",
    }
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print(
        json.dumps(
            {
                "N": n,
                "has_open_think": c["has_open_think"],
                "has_open_answer": c["has_open_answer"],
                "has_boxed": c["has_boxed"],
                "start_close_think": c["start_close_think"],
                "start_close_answer": c["start_close_answer"],
                "only_close_answer": c["only_close_answer"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
