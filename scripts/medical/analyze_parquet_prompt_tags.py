#!/usr/bin/env python3
"""Analyze prompt tag patterns in parquet SFT file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prompt tags in parquet.")
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    arr = pq.read_table(args.parquet, columns=["prompt"]).column("prompt").to_pylist()
    n = len(arr)
    c: Counter[str] = Counter()
    for s in arr:
        t = (s or "").strip()
        c["contains_open_think"] += "<think>" in t
        c["contains_close_think"] += "</think>" in t
        c["contains_open_answer"] += "<answer>" in t
        c["contains_close_answer"] += "</answer>" in t
        c["ends_with_open_think"] += t.endswith("<think>")
        c["ends_with_close_think"] += t.endswith("</think>")
        c["ends_with_open_answer"] += t.endswith("<answer>")
        c["ends_with_close_answer"] += t.endswith("</answer>")

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
                "contains_open_think": c["contains_open_think"],
                "contains_close_think": c["contains_close_think"],
                "contains_open_answer": c["contains_open_answer"],
                "ends_with_open_think": c["ends_with_open_think"],
                "ends_with_open_answer": c["ends_with_open_answer"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
