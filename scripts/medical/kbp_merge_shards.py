#!/usr/bin/env python3
"""Merge KBP shard runs into a final run directory.

This script copies base run files (optional), appends shard response/parsed parquet
parts with fresh indices, and validates there are no duplicated
(question_id, probe_index) rows in parsed outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge KBP shard outputs.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="medqa_grpo_train")
    parser.add_argument("--final_run_id", type=str, required=True)
    parser.add_argument("--base_run_id", type=str, default="")
    parser.add_argument("--shard_run_ids", type=str, required=True)
    parser.add_argument("--overwrite_final", action="store_true")
    return parser.parse_args()


def _existing_part_index_max(split_dir: Path) -> int:
    max_idx = -1
    for p in split_dir.glob("part-*.parquet"):
        try:
            idx = int(p.stem.split("-")[-1])
        except Exception:
            continue
        max_idx = max(max_idx, idx)
    return max_idx


def _copy_parts(src_split_dir: Path, dst_split_dir: Path) -> int:
    files = sorted(src_split_dir.glob("part-*.parquet"))
    if not files:
        return 0
    next_idx = _existing_part_index_max(dst_split_dir) + 1
    copied = 0
    for src in files:
        dst = dst_split_dir / f"part-{next_idx:05d}.parquet"
        shutil.copy2(src, dst)
        next_idx += 1
        copied += 1
    return copied


def _validate_no_dup_probe(parsed_split_dir: Path) -> dict[str, int]:
    seen: set[tuple[str, int]] = set()
    row_count = 0
    duplicate_count = 0
    for p in sorted(parsed_split_dir.glob("part-*.parquet")):
        t = pq.ParquetFile(p).read(columns=["question_id", "probe_index"])
        qids = t.column("question_id").to_pylist()
        probes = t.column("probe_index").to_pylist()
        for qid, probe in zip(qids, probes):
            row_count += 1
            key = (str(qid), int(probe))
            if key in seen:
                duplicate_count += 1
            else:
                seen.add(key)
    if duplicate_count > 0:
        raise RuntimeError(f"duplicate (question_id,probe_index) detected: {duplicate_count}")
    return {"parsed_rows": int(row_count), "unique_pairs": int(len(seen))}


def main() -> None:
    args = parse_args()

    run_root = args.output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    final_run_dir = run_root / args.final_run_id
    if final_run_dir.exists() and not args.overwrite_final:
        raise FileExistsError(f"final run dir exists, pass --overwrite_final: {final_run_dir}")

    if final_run_dir.exists():
        shutil.rmtree(final_run_dir)

    if args.base_run_id:
        base_run_dir = run_root / args.base_run_id
        if not base_run_dir.exists():
            raise FileNotFoundError(f"base run dir not found: {base_run_dir}")
        shutil.copytree(base_run_dir, final_run_dir)
    else:
        (final_run_dir / "responses").mkdir(parents=True, exist_ok=True)
        (final_run_dir / "parsed").mkdir(parents=True, exist_ok=True)
        (final_run_dir / "labels").mkdir(parents=True, exist_ok=True)
        (final_run_dir / "reports").mkdir(parents=True, exist_ok=True)

    responses_split_dir = final_run_dir / "responses" / f"split={args.split}"
    parsed_split_dir = final_run_dir / "parsed" / f"split={args.split}"
    responses_split_dir.mkdir(parents=True, exist_ok=True)
    parsed_split_dir.mkdir(parents=True, exist_ok=True)

    shard_run_ids = [x.strip() for x in args.shard_run_ids.replace(":", ",").split(",") if x.strip()]
    if not shard_run_ids:
        raise ValueError("--shard_run_ids parsed empty")

    shard_stats: dict[str, dict[str, int]] = {}
    for rid in shard_run_ids:
        shard_dir = run_root / rid
        if not shard_dir.exists():
            raise FileNotFoundError(f"shard run dir not found: {shard_dir}")
        src_resp = shard_dir / "responses" / f"split={args.split}"
        src_parsed = shard_dir / "parsed" / f"split={args.split}"
        if not src_resp.exists() or not src_parsed.exists():
            raise FileNotFoundError(f"shard split dir missing: {rid}")

        moved_resp = _copy_parts(src_resp, responses_split_dir)
        moved_parsed = _copy_parts(src_parsed, parsed_split_dir)
        shard_stats[rid] = {
            "copied_response_parts": int(moved_resp),
            "copied_parsed_parts": int(moved_parsed),
        }

    validation = _validate_no_dup_probe(parsed_split_dir)

    report = {
        "final_run_id": args.final_run_id,
        "base_run_id": args.base_run_id,
        "split": args.split,
        "shard_run_ids": shard_run_ids,
        "shard_stats": shard_stats,
        "validation": validation,
        "responses_parts_total": len(list(responses_split_dir.glob("part-*.parquet"))),
        "parsed_parts_total": len(list(parsed_split_dir.glob("part-*.parquet"))),
    }

    report_path = final_run_dir / "reports" / "kbp_merge_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"status": "ok", "report": str(report_path), **validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
