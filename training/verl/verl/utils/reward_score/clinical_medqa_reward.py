# Copyright 2026
#
# Stage D reward for medical MCQ training.

from __future__ import annotations

import re
from typing import Any

_BOXED_PATTERN = re.compile(r"\\boxed\s*{(.*?)}", re.DOTALL)
_OPTION_LABEL_PATTERN = re.compile(r"^\s*([A-Za-z])\s*[:.)\]-]?\s*")
_THINK_ANSWER_PATTERN = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_single_boxed(solution_str: str) -> tuple[bool, str | None]:
    """Return whether exactly one boxed answer exists and its content."""
    if not isinstance(solution_str, str):
        return False, None
    matches = _BOXED_PATTERN.findall(solution_str)
    if len(matches) != 1:
        return False, None
    return True, matches[0].strip()


def normalize_choice(text: str) -> str:
    """Normalize a MCQ choice token to an uppercase option label when possible."""
    if not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    match = _OPTION_LABEL_PATTERN.match(stripped)
    if match:
        return match.group(1).upper()

    compact = re.sub(r"\s+", "", stripped).upper()
    if len(compact) == 1 and "A" <= compact <= "Z":
        return compact
    return ""


def is_abstain_boxed(text: str) -> bool:
    """Only boxed abstention should be treated as abstain."""
    if not isinstance(text, str):
        return False
    canonical = re.sub(r"[^a-z0-9]+", "", text.lower())
    return canonical in {"idontknow", "idonotknow", "idk"}


def _extract_target_set(ground_truth: dict[str, Any]) -> set[str]:
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list):
        return set()
    return {normalize_choice(x) for x in targets if normalize_choice(x)}


def _extract_choice_set(ground_truth: dict[str, Any]) -> set[str]:
    choices = ground_truth.get("choices", [])
    if not isinstance(choices, list):
        return set()

    labels: set[str] = set()
    for item in choices:
        if not isinstance(item, str):
            continue
        m = _OPTION_LABEL_PATTERN.match(item)
        if m:
            labels.add(m.group(1).upper())
    return labels


def compute_outcome_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    plain_ternary: bool = True,
) -> dict[str, Any]:
    """Compute plain ternary score (+1 / 0 / -1) for medical MCQ."""
    valid_boxed, boxed_content = extract_single_boxed(solution_str)
    if not valid_boxed or boxed_content is None:
        return {
            "outcome_score": -1.0,
            "is_abstain": 0,
            "is_boxed_valid": 0,
            "prediction_type": "parse_fail",
        }

    if is_abstain_boxed(boxed_content):
        # Stage D-1 default: plain ternary, abstain=0.
        if plain_ternary:
            outcome = 0.0
        else:
            # Knowledge-enhanced mode (kept for explicit future use only).
            is_ook = bool(ground_truth.get("out_of_knowledge", False))
            outcome = 1.0 if is_ook else 0.0
        return {
            "outcome_score": outcome,
            "is_abstain": 1,
            "is_boxed_valid": 1,
            "prediction_type": "abstain",
        }

    normalized_pred = normalize_choice(boxed_content)
    if not normalized_pred:
        return {
            "outcome_score": -1.0,
            "is_abstain": 0,
            "is_boxed_valid": 0,
            "prediction_type": "wrong",
        }

    target_set = _extract_target_set(ground_truth)
    choice_set = _extract_choice_set(ground_truth)

    if not plain_ternary and bool(ground_truth.get("out_of_knowledge", False)):
        return {
            "outcome_score": -1.0,
            "is_abstain": 0,
            "is_boxed_valid": int(normalized_pred in choice_set) if choice_set else 1,
            "prediction_type": "wrong",
        }

    if normalized_pred in target_set:
        return {
            "outcome_score": 1.0,
            "is_abstain": 0,
            "is_boxed_valid": 1,
            "prediction_type": "correct",
        }

    is_valid = int(normalized_pred in choice_set) if choice_set else 1
    return {
        "outcome_score": -1.0,
        "is_abstain": 0,
        "is_boxed_valid": is_valid,
        "prediction_type": "wrong",
    }


def compute_format_reward(solution_str: str) -> float:
    """
    Reward format contract:
    - +1.0 if <think>...</think><answer>...</answer> exists in order.
    - +0.5 if answer section includes explanatory text plus one boxed answer.
    """
    if not isinstance(solution_str, str):
        return 0.0

    match = _THINK_ANSWER_PATTERN.search(solution_str.strip())
    if not match:
        return 0.0

    score = 1.0
    answer_section = match.group(2).strip()
    valid_boxed, _ = extract_single_boxed(answer_section)
    if valid_boxed:
        boxed_only = _BOXED_PATTERN.sub("", answer_section).strip()
        if boxed_only:
            score += 0.5
    return score


def compute_consistency_reward(solution_str: str, ground_truth: dict[str, Any]) -> float:
    """
    Reward consistency between answer narrative and boxed answer.
    Returns 0.0 or 1.0.
    """
    valid_boxed, boxed_content = extract_single_boxed(solution_str)
    if not valid_boxed or boxed_content is None:
        return 0.0

    if is_abstain_boxed(boxed_content):
        lower = solution_str.lower()
        return 1.0 if ("don't know" in lower or "dont know" in lower or "do not know" in lower) else 0.0

    boxed_choice = normalize_choice(boxed_content)
    if not boxed_choice:
        return 0.0

    match = _THINK_ANSWER_PATTERN.search(solution_str.strip())
    answer_section = match.group(2) if match else solution_str
    extracted = re.findall(r"\b([A-Z])\b", answer_section.upper())

    choice_set = _extract_choice_set(ground_truth)
    mentions = [x for x in extracted if not choice_set or x in choice_set]
    unique_mentions = sorted(set(mentions))

    if len(unique_mentions) == 1 and unique_mentions[0] == boxed_choice:
        return 1.0
    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """NaiveRewardManager-compatible reward entry for Stage D."""
    del data_source, extra_info  # kept for signature compatibility

    stage_mode = str(kwargs.get("stage_mode", "d1")).strip().lower()
    is_d3_hard_ook = stage_mode in {"d3", "d3_hard_ook", "hard_ook"}
    k = float(kwargs.get("k", 1.0))
    plain_ternary = bool(kwargs.get("plain_ternary", not is_d3_hard_ook))
    enable_format = bool(kwargs.get("enable_format", stage_mode == "d2"))
    enable_consistency = bool(kwargs.get("enable_consistency", stage_mode == "d2"))
    lambda_format = float(kwargs.get("lambda_format", 1.0))
    lambda_consistency = float(kwargs.get("lambda_consistency", 0.5))

    outcome = compute_outcome_score(solution_str=solution_str, ground_truth=ground_truth, plain_ternary=plain_ternary)
    format_score = compute_format_reward(solution_str) if enable_format else 0.0
    consistency_score = compute_consistency_reward(solution_str, ground_truth) if enable_consistency else 0.0

    final_score = (
        k * float(outcome["outcome_score"])
        + lambda_format * float(format_score)
        + lambda_consistency * float(consistency_score)
    )

    prediction_type_id = {
        "parse_fail": 0,
        "abstain": 1,
        "wrong": 2,
        "correct": 3,
    }.get(str(outcome.get("prediction_type", "")), -1)
    stage_mode_id = 1 if stage_mode == "d1" else 2 if stage_mode == "d2" else 3 if is_d3_hard_ook else 0

    return {
        "score": float(final_score),
        "outcome_score": float(outcome["outcome_score"]),
        "format_score": float(format_score),
        "consistency_score": float(consistency_score),
        "is_abstain": int(outcome["is_abstain"]),
        "is_boxed_valid": int(outcome["is_boxed_valid"]),
        "prediction_type": int(prediction_type_id),
        "stage_mode": int(stage_mode_id),
        "is_out_of_knowledge": int(bool(ground_truth.get("out_of_knowledge", False))),
        "plain_ternary": int(bool(plain_ternary)),
        "k": k,
        "lambda_format": lambda_format,
        "lambda_consistency": lambda_consistency,
    }
