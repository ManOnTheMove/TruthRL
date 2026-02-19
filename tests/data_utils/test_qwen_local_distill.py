from __future__ import annotations

import csv
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.medical import qwen_local_distill


def test_select_shard_rows_and_sample_limit() -> None:
    rows = [{"question_id": f"q{i}", "question_index": i, "problem": "p", "target": ["A"]} for i in range(10)]
    shard_rows = qwen_local_distill.select_shard_rows(rows, shard_index=1, num_shards=3, sample_limit=2)
    assert [x["question_id"] for x in shard_rows] == ["q1", "q4"]


def test_normalize_completion_format_with_force_target() -> None:
    raw = "This is a short explanation without tags."
    normalized = qwen_local_distill.normalize_completion_format(raw, target_choice="C", force_target_boxed=True)
    ok, reason = qwen_local_distill.validate_completion_contract(normalized)
    assert ok, reason
    assert "\\boxed{C}" in normalized


def test_validate_completion_contract_negative() -> None:
    bad = "<think>abc</think><answer>no boxed here</answer>"
    ok, reason = qwen_local_distill.validate_completion_contract(bad)
    assert not ok
    assert reason == "boxed_missing"


def test_normalize_completion_avoids_nested_tags() -> None:
    raw = "<think>Reasoning text</think> Final answer is \\\\boxed{A}."
    normalized = qwen_local_distill.normalize_completion_format(raw, target_choice="A", force_target_boxed=True)
    assert normalized.count("<think>") == 1
    assert normalized.count("</think>") == 1
    assert normalized.count("<answer>") == 1
    assert normalized.count("</answer>") == 1
    ok, reason = qwen_local_distill.validate_completion_contract(normalized)
    assert ok, reason


def test_load_sft_source_csv(tmp_path: Path) -> None:
    src = tmp_path / "sft.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question_index", "problem", "target"])
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "qid_1",
                "question_index": 3,
                "problem": "Question text",
                "target": json.dumps(["B"]),
            }
        )

    rows = qwen_local_distill.load_sft_source_csv(src)
    assert len(rows) == 1
    assert rows[0]["question_id"] == "qid_1"
    assert rows[0]["question_index"] == 3
    assert rows[0]["target"] == ["B"]


def test_write_completion_atomic(tmp_path: Path) -> None:
    out = tmp_path / "distill" / "qid.txt"
    qwen_local_distill.write_completion_atomic(out, "abc")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "abc"
