# Copyright 2026
#
# Stage D reward for medical MCQ training.

from __future__ import annotations

import re
from typing import Any

_BOXED_PATTERN = re.compile(r"\\boxed\s*{(.*?)}", re.DOTALL)
_OPTION_LABEL_PATTERN = re.compile(r"^\s*([A-Za-z])\s*[:.)\]-]?\s*")


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
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return normalized in {"i dont know", "i do not know", "idk"}


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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """NaiveRewardManager-compatible reward entry for Stage D."""
    del data_source, extra_info  # kept for signature compatibility

    k = float(kwargs.get("k", 1.0))
    plain_ternary = bool(kwargs.get("plain_ternary", True))

    outcome = compute_outcome_score(solution_str=solution_str, ground_truth=ground_truth, plain_ternary=plain_ternary)
    final_score = k * float(outcome["outcome_score"])

    return {
        "score": float(final_score),
        "outcome_score": float(outcome["outcome_score"]),
        "is_abstain": int(outcome["is_abstain"]),
        "is_boxed_valid": int(outcome["is_boxed_valid"]),
        "prediction_type": outcome["prediction_type"],
        "k": k,
    }
