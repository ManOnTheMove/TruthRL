from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.medical import build_distill_sft, build_medmcqa, build_medqa, validate_medical_data


def test_medqa_parser_and_split_reproducibility(tmp_path: Path) -> None:
    raw_path = tmp_path / "medqa_train.jsonl"
    record = {
        "question": "Q?",
        "answer": "Option A",
        "options": {"A": "Option A", "B": "Option B"},
        "answer_idx": "A",
    }
    raw_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = build_medqa.load_medqa_split(raw_path)
    assert len(loaded) == 1
    normalized = build_medqa.normalize_medqa_record(loaded[0], split="train", index=0)
    assert normalized["target"] == ["A"]
    assert "A: Option A" in normalized["choices"]

    sample_records = [dict(normalized, question_id=f"id_{i}", question_index=i) for i in range(20)]
    sft_1, grpo_1 = build_medqa.split_for_sft_grpo(sample_records, policy="half", seed=42)
    sft_2, grpo_2 = build_medqa.split_for_sft_grpo(sample_records, policy="half", seed=42)
    assert [x["question_id"] for x in sft_1] == [x["question_id"] for x in sft_2]
    assert [x["question_id"] for x in grpo_1] == [x["question_id"] for x in grpo_2]
    assert set(x["question_id"] for x in sft_1).isdisjoint(set(x["question_id"] for x in grpo_1))


def test_medmcqa_parser_independent_and_bad_json_error(tmp_path: Path) -> None:
    good = {
        "question": "MCQ?",
        "opa": "A1",
        "opb": "B1",
        "opc": "C1",
        "opd": "D1",
        "cop": 2,
        "id": "qid_1",
        "choice_type": "single",
    }
    good_path = tmp_path / "medmcqa_dev.json"
    good_path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    rows = build_medmcqa.load_medmcqa_split(good_path)
    norm = build_medmcqa.normalize_medmcqa_record(rows[0], split="dev", index=0)
    assert norm["target"] == ["B"]
    assert norm["question_id"] == "qid_1"

    bad_path = tmp_path / "medmcqa_bad.json"
    bad_path.write_text("{this is not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_medmcqa.load_medmcqa_split(bad_path)


def test_to_rl_record_and_schema_validation(tmp_path: Path) -> None:
    medqa_example = {
        "question_id": "medqa_train_0",
        "question_index": 0,
        "question": "Q",
        "problem": "Q\nA: x\nB: y",
        "choice_type": "single",
        "choices": ["A: x", "B: y"],
        "target": ["A"],
        "source_split": "train",
    }
    rl_row = build_medqa.to_rl_record(medqa_example, data_source="medqa", split="train_grpo_half", index=0)
    assert set(rl_row.keys()) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert rl_row["reward_model"]["ground_truth"]["out_of_knowledge"] is False
    assert "\\boxed{" in rl_row["prompt"][0]["content"]

    parquet_path = tmp_path / "medqa_rl.parquet"
    build_medqa._write_parquet([rl_row], parquet_path)
    summary = validate_medical_data.validate_rl_schema(parquet_path)
    assert summary.row_count == 1
    assert summary.empty_target_count == 0
    assert summary.prompt_boxed_contract_violations == 0


def test_to_sft_record_and_sft_schema_validation(tmp_path: Path) -> None:
    example = {"problem": "P", "target": ["A"], "question_id": "qid", "question_index": 0}
    row = build_distill_sft.to_sft_record(example, completion="<answer>\\boxed{A}</answer>")
    assert set(row.keys()) == {"prompt", "response"}
    assert "\\boxed{" in row["prompt"]

    parquet_path = tmp_path / "medqa_sft.parquet"
    build_distill_sft._write_parquet([row], parquet_path)
    summary = validate_medical_data.validate_sft_schema(parquet_path)
    assert summary.row_count == 1


def test_missing_completion_strict_mode_fails_and_reports(tmp_path: Path) -> None:
    sft_csv = tmp_path / "medqa_train_sft_half.csv"
    with sft_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question_index", "problem", "target"])
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "qid_1",
                "question_index": 1,
                "problem": "Question text",
                "target": json.dumps(["A"]),
            }
        )
    distill_dir = tmp_path / "distill"
    distill_dir.mkdir()
    out_dir = tmp_path / "out"
    reports_dir = tmp_path / "reports"

    script = REPO_ROOT / "data_utils" / "medical" / "build_distill_sft.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sft_csv",
            str(sft_csv),
            "--distill_dir",
            str(distill_dir),
            "--output_dir",
            str(out_dir),
                "--reports_dir",
                str(reports_dir),
                "--missing_report_path",
                str(reports_dir / "missing_distill_completions.txt"),
                "--require_distill",
            ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Missing distilled completions detected" in (proc.stderr + proc.stdout)
    assert (reports_dir / "missing_distill_completions.txt").exists()


def test_update_ook_labels(tmp_path: Path) -> None:
    base_example = {
        "question": "Q",
        "problem": "Q\nA: x\nB: y",
        "choice_type": "single",
        "choices": ["A: x", "B: y"],
        "target": ["A"],
        "source_split": "train",
    }
    rows = []
    for i in range(2):
        item = dict(base_example, question_id=f"qid_{i}", question_index=i)
        rows.append(build_medqa.to_rl_record(item, data_source="medqa", split="train_grpo_half", index=i))

    in_parquet = tmp_path / "input.parquet"
    out_parquet = tmp_path / "output.parquet"
    build_medqa._write_parquet(rows, in_parquet)

    ook_json = tmp_path / "ook.json"
    ook_json.write_text(json.dumps({"qid_0": True, "qid_missing": True}), encoding="utf-8")

    summary = validate_medical_data.update_ook_labels(in_parquet, ook_json, out_parquet)
    assert summary["updated_rows"] == 1
    assert summary["mapping_question_ids_missing_in_parquet"] == 1

    validated = validate_medical_data.validate_rl_schema(out_parquet)
    assert validated.out_of_knowledge_true_count == 1
