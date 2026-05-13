#!/usr/bin/env python3
"""Stage D2.6: build OOK mapping from KBP labels and update RL parquet.

This tool performs two D2.6 actions in one pass:
1) Backfill reward_model.ground_truth.out_of_knowledge from KBP labels.
2) Upgrade prompt contract to allow \boxed{I don't know} as a valid abstain action.

It never mutates input parquet in-place; output is written to a new parquet file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="D2.6 OOK backfill + prompt contract upgrade")
    parser.add_argument(
        "--kbp-labels",
        type=Path,
        required=True,
        help="Path to labels/kbp_labels_by_question.parquet from canonical D2.5 run.",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=repo_root / "data" / "medical" / "verl" / "medqa_grpo_train.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=repo_root / "data" / "medical" / "verl" / "medqa_grpo_train_with_ook_d26.parquet",
    )
    parser.add_argument(
        "--ook-json",
        type=Path,
        default=None,
        help="Optional path to dump question_id -> out_of_knowledge mapping json.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to write D2.6 summary json (e.g., reports/ook_update_summary.json).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "zstd", "gzip", "brotli", "lz4", "none"],
    )
    return parser.parse_args()


OLD_SENTENCE = r"The final selected option MUST appear in \boxed{} (for example, \boxed{A})."

NEW_BLOCK = (
    "The final boxed action MUST be exactly one of:\n"
    "- Option answer: \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}\n"
    "- Abstention when knowledge is insufficient: \\boxed{I don't know}\n\n"
    "Do not output any boxed content other than the final boxed action."
)


def _upgrade_prompt_contract(prompt_obj: Any) -> tuple[Any, bool, bool]:
    """Return (new_prompt, changed, has_idk_after)."""
    if not isinstance(prompt_obj, list) or not prompt_obj:
        return prompt_obj, False, False
    first = prompt_obj[0]
    if not isinstance(first, dict):
        return prompt_obj, False, False
    content = first.get("content")
    if not isinstance(content, str):
        return prompt_obj, False, False

    has_idk_before = "\\boxed{I don't know}" in content or "I don't know" in content
    new_content = content

    if OLD_SENTENCE in content:
        new_content = content.replace(OLD_SENTENCE, NEW_BLOCK)
    elif not has_idk_before:
        marker = "\n\nQuestion:\n"
        insert = "\n\n" + NEW_BLOCK
        if marker in content:
            new_content = content.replace(marker, insert + marker, 1)
        else:
            new_content = content + insert

    changed = new_content != content
    if changed:
        first = dict(first)
        first["content"] = new_content
        new_prompt = list(prompt_obj)
        new_prompt[0] = first
        return new_prompt, True, ("\\boxed{I don't know}" in new_content or "I don't know" in new_content)

    return prompt_obj, False, ("\\boxed{I don't know}" in content or "I don't know" in content)


def _load_ook_map(labels_path: Path) -> dict[str, bool]:
    if not labels_path.exists():
        raise FileNotFoundError(f"KBP labels not found: {labels_path}")
    table = pq.read_table(labels_path, columns=["question_id", "ook"])
    rows = table.to_pylist()
    out: dict[str, bool] = {}
    for r in rows:
        qid = str(r.get("question_id"))
        if qid in out:
            raise ValueError(f"Duplicate question_id in labels: {qid}")
        out[qid] = bool(r.get("ook", False))
    return out


def _targets_digest(rows: list[dict[str, Any]]) -> str:
    pairs: list[tuple[str, Any]] = []
    for row in rows:
        extra_info = row.get("extra_info") or {}
        reward_model = row.get("reward_model") or {}
        gt = reward_model.get("ground_truth") or {}
        qid = str(extra_info.get("question_id", ""))
        pairs.append((qid, gt.get("target")))
    pairs.sort(key=lambda x: x[0])
    payload = json.dumps(pairs, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"input parquet not found: {args.input_parquet}")

    ook_map = _load_ook_map(args.kbp_labels)

    input_table = pq.read_table(args.input_parquet)
    rows = input_table.to_pylist()

    targets_digest_before = _targets_digest(rows)

    updated_ook_rows = 0
    existing_ook_true_before = 0
    prompt_upgraded_rows = 0
    prompt_has_idk_after = 0
    missing_in_labels = 0

    seen_qids: set[str] = set()
    for i, row in enumerate(rows):
        extra_info = row.get("extra_info")
        reward_model = row.get("reward_model")
        prompt = row.get("prompt")
        if not isinstance(extra_info, dict) or not isinstance(reward_model, dict):
            raise ValueError(f"row {i}: invalid extra_info/reward_model")

        qid = extra_info.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"row {i}: invalid question_id")
        seen_qids.add(qid)

        gt = reward_model.get("ground_truth")
        if not isinstance(gt, dict):
            raise ValueError(f"row {i}: invalid ground_truth")

        old_ook = bool(gt.get("out_of_knowledge", False))
        if old_ook:
            existing_ook_true_before += 1

        if qid in ook_map:
            new_ook = bool(ook_map[qid])
            if new_ook != old_ook:
                updated_ook_rows += 1
            gt["out_of_knowledge"] = new_ook
        else:
            missing_in_labels += 1

        new_prompt, changed, has_idk = _upgrade_prompt_contract(prompt)
        if changed:
            row["prompt"] = new_prompt
            prompt_upgraded_rows += 1
        if has_idk:
            prompt_has_idk_after += 1

    missing_in_parquet = sum(1 for qid in ook_map if qid not in seen_qids)

    targets_digest_after = _targets_digest(rows)
    target_unchanged = targets_digest_before == targets_digest_after
    if not target_unchanged:
        raise RuntimeError("ground_truth.target changed unexpectedly; aborting")

    output_table = pa.Table.from_pylist(rows)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    pq.write_table(output_table, args.output_parquet, compression=compression)

    # Re-open for post-write checks.
    out_rows = pq.read_table(args.output_parquet, columns=["reward_model", "prompt", "extra_info"]).to_pylist()
    out_ook_true = 0
    out_prompt_has_idk = 0
    for row in out_rows:
        gt = (row.get("reward_model") or {}).get("ground_truth") or {}
        if bool(gt.get("out_of_knowledge", False)):
            out_ook_true += 1
        prompt = row.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            content = prompt[0].get("content")
            if isinstance(content, str) and ("\\boxed{I don't know}" in content or "I don't know" in content):
                out_prompt_has_idk += 1

    if args.ook_json is not None:
        args.ook_json.parent.mkdir(parents=True, exist_ok=True)
        args.ook_json.write_text(json.dumps(ook_map, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "stage": "D2.6",
        "status": "pass",
        "kbp_labels": str(args.kbp_labels),
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "ook_json": str(args.ook_json) if args.ook_json is not None else None,
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "mapping_size": len(ook_map),
        "existing_ook_true_before": int(existing_ook_true_before),
        "updated_ook_rows": int(updated_ook_rows),
        "output_ook_true_count": int(out_ook_true),
        "missing_question_ids_in_labels": int(missing_in_labels),
        "mapping_question_ids_missing_in_parquet": int(missing_in_parquet),
        "prompt_upgraded_rows": int(prompt_upgraded_rows),
        "output_prompt_has_idk_rows": int(out_prompt_has_idk),
        "target_unchanged": bool(target_unchanged),
        "compression": args.compression,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
