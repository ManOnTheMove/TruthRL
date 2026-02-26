# Copyright 2026

from verl.utils.reward_score.clinical_medqa_reward import (
    compute_outcome_score,
    compute_score,
    extract_single_boxed,
    is_abstain_boxed,
    normalize_choice,
)


def _gt(target="A", out_of_knowledge=False):
    return {
        "target": [target],
        "choices": [
            "A: first",
            "B: second",
            "C: third",
            "D: fourth",
            "E: fifth",
        ],
        "out_of_knowledge": out_of_knowledge,
    }


def test_extract_single_boxed_ok():
    ok, content = extract_single_boxed("x \\boxed{A} y")
    assert ok is True
    assert content == "A"


def test_extract_single_boxed_none():
    ok, content = extract_single_boxed("no boxed here")
    assert ok is False
    assert content is None


def test_extract_single_boxed_multiple():
    ok, content = extract_single_boxed("\\boxed{A} and \\boxed{B}")
    assert ok is False
    assert content is None


def test_normalize_choice():
    assert normalize_choice("a") == "A"
    assert normalize_choice(" A ) ") == "A"
    assert normalize_choice("B: answer") == "B"
    assert normalize_choice("not-an-option") == ""


def test_is_abstain_boxed():
    assert is_abstain_boxed("I don't know") is True
    assert is_abstain_boxed("i dont know") is True
    assert is_abstain_boxed("A") is False


def test_compute_score_correct():
    res = compute_score(
        data_source="medqa",
        solution_str="<answer>\\boxed{A}</answer>",
        ground_truth=_gt("A"),
    )
    assert res["score"] == 1.0
    assert res["outcome_score"] == 1.0
    assert res["prediction_type"] == "correct"
    assert res["is_boxed_valid"] == 1
    assert res["is_abstain"] == 0


def test_compute_score_wrong_option():
    res = compute_score(
        data_source="medqa",
        solution_str="<answer>\\boxed{B}</answer>",
        ground_truth=_gt("A"),
    )
    assert res["score"] == -1.0
    assert res["prediction_type"] == "wrong"
    assert res["is_boxed_valid"] == 1


def test_compute_score_abstain():
    res = compute_score(
        data_source="medqa",
        solution_str="<answer>\\boxed{I don't know}</answer>",
        ground_truth=_gt("A"),
    )
    assert res["score"] == 0.0
    assert res["outcome_score"] == 0.0
    assert res["prediction_type"] == "abstain"
    assert res["is_abstain"] == 1


def test_compute_score_no_boxed_is_parse_fail():
    res = compute_score(
        data_source="medqa",
        solution_str="answer is A",
        ground_truth=_gt("A"),
    )
    assert res["score"] == -1.0
    assert res["prediction_type"] == "parse_fail"
    assert res["is_boxed_valid"] == 0


def test_compute_score_multiple_boxed_is_parse_fail():
    res = compute_score(
        data_source="medqa",
        solution_str="\\boxed{A} ... \\boxed{B}",
        ground_truth=_gt("A"),
    )
    assert res["score"] == -1.0
    assert res["prediction_type"] == "parse_fail"
    assert res["is_boxed_valid"] == 0


def test_compute_score_illegal_choice_is_wrong_and_not_boxed_valid():
    res = compute_score(
        data_source="medqa",
        solution_str="\\boxed{Z}",
        ground_truth=_gt("A"),
    )
    assert res["score"] == -1.0
    assert res["prediction_type"] == "wrong"
    assert res["is_boxed_valid"] == 0


def test_compute_score_k_scaling():
    res = compute_score(
        data_source="medqa",
        solution_str="\\boxed{A}",
        ground_truth=_gt("A"),
        k=3,
    )
    assert res["outcome_score"] == 1.0
    assert res["score"] == 3.0
    assert res["k"] == 3.0


def test_compute_outcome_score_knowledge_enhanced_for_ook():
    abstain = compute_outcome_score(
        solution_str="\\boxed{I don't know}",
        ground_truth=_gt("A", out_of_knowledge=True),
        plain_ternary=False,
    )
    answer = compute_outcome_score(
        solution_str="\\boxed{A}",
        ground_truth=_gt("A", out_of_knowledge=True),
        plain_ternary=False,
    )
    assert abstain["outcome_score"] == 1.0
    assert answer["outcome_score"] == -1.0
