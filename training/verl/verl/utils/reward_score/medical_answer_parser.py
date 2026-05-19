# Copyright 2026
#
# Shared answer parser for EMBC medical MCQ reward, KBP, and eval paths.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


POLICY_SINGLE_BOXED_STRICT = "single_boxed_strict"
POLICY_FINAL_ANSWER_LAST_BOXED = "final_answer_last_boxed"
POLICY_FIRST_BOXED_LEGACY = "first_boxed_legacy"

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
OPTION_LABEL_PATTERN = re.compile(r"^\s*([A-Za-z])\s*(?:$|[).:]|-\s+)")


@dataclass(frozen=True)
class ParsedMedicalAnswer:
    answer: str
    boxed_raw: str
    normalized_choice: str
    parse_status: str
    policy: str
    boxed_count: int
    used_answer_block: bool
    is_abstain: bool
    is_valid_choice: bool


def find_boxed_answers(text: str) -> list[str]:
    """Return balanced contents of every LaTeX \\boxed{...} in order."""
    if not isinstance(text, str):
        return []

    answers: list[str] = []
    search_from = 0
    while True:
        start = text.find("\\boxed", search_from)
        if start < 0:
            break

        cursor = start + len("\\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            search_from = cursor
            continue

        depth = 0
        content_start = cursor + 1
        idx = cursor
        while idx < len(text):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[content_start:idx].strip())
                    search_from = idx + 1
                    break
            idx += 1
        else:
            break

    return answers


def extract_single_boxed(solution_str: str) -> tuple[bool, str | None]:
    """Return whether exactly one balanced boxed answer exists and its content."""
    boxed_values = find_boxed_answers(solution_str)
    if len(boxed_values) != 1:
        return False, None
    return True, boxed_values[0]


def normalize_choice(text: str) -> str:
    """Normalize explicit MCQ labels like A, A), A., A:, or A - text."""
    if not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()

    match = OPTION_LABEL_PATTERN.match(stripped)
    if match:
        return match.group(1).upper()

    return ""


def is_abstain_boxed(text: str) -> bool:
    """Only boxed abstention text should be treated as abstain."""
    if not isinstance(text, str):
        return False
    canonical = re.sub(r"[^a-z0-9]+", "", text.lower())
    return canonical in {"idontknow", "idonotknow", "idk"}


def _normalize_choice_set(choice_set: Iterable[str] | None) -> set[str]:
    if choice_set is None:
        return set()
    return {value for value in (normalize_choice(str(item)) for item in choice_set) if value}


def _selected_boxed(
    text: str,
    policy: str,
) -> tuple[str | None, int, bool, str]:
    global_boxed = find_boxed_answers(text)
    boxed_count = len(global_boxed)

    if policy == POLICY_SINGLE_BOXED_STRICT:
        if boxed_count == 0:
            return None, boxed_count, False, "no_boxed"
        if boxed_count != 1:
            return " | ".join(global_boxed), boxed_count, False, "multi_boxed"
        return global_boxed[0], boxed_count, False, "ok"

    if policy == POLICY_FINAL_ANSWER_LAST_BOXED:
        answer_blocks = ANSWER_TAG_PATTERN.findall(text if isinstance(text, str) else "")
        if answer_blocks:
            answer_boxed = find_boxed_answers(answer_blocks[-1])
            if answer_boxed:
                return answer_boxed[-1], boxed_count, True, "ok"
        if global_boxed:
            return global_boxed[-1], boxed_count, False, "ok"
        return None, boxed_count, False, "no_boxed"

    if policy == POLICY_FIRST_BOXED_LEGACY:
        if global_boxed:
            return global_boxed[0], boxed_count, False, "ok"
        return None, boxed_count, False, "no_boxed"

    raise ValueError(f"unknown medical answer parser policy: {policy}")


def parse_medical_answer(
    text: str,
    *,
    policy: str,
    choice_set: Iterable[str] | None = None,
) -> ParsedMedicalAnswer:
    """Parse a medical MCQ answer with an explicit compatibility policy."""
    selected, boxed_count, used_answer_block, status = _selected_boxed(text if isinstance(text, str) else "", policy)
    if status in {"no_boxed", "multi_boxed"}:
        boxed_raw = "" if selected is None else selected
        return ParsedMedicalAnswer(
            answer=boxed_raw,
            boxed_raw=boxed_raw,
            normalized_choice="",
            parse_status=status,
            policy=policy,
            boxed_count=boxed_count,
            used_answer_block=used_answer_block,
            is_abstain=False,
            is_valid_choice=False,
        )

    boxed_raw = selected or ""
    if is_abstain_boxed(boxed_raw):
        return ParsedMedicalAnswer(
            answer=boxed_raw,
            boxed_raw=boxed_raw,
            normalized_choice="I don't know",
            parse_status="ok_abstain",
            policy=policy,
            boxed_count=boxed_count,
            used_answer_block=used_answer_block,
            is_abstain=True,
            is_valid_choice=False,
        )

    normalized = normalize_choice(boxed_raw)
    if not normalized:
        return ParsedMedicalAnswer(
            answer=boxed_raw,
            boxed_raw=boxed_raw,
            normalized_choice="",
            parse_status="invalid_choice",
            policy=policy,
            boxed_count=boxed_count,
            used_answer_block=used_answer_block,
            is_abstain=False,
            is_valid_choice=False,
        )

    normalized_choices = _normalize_choice_set(choice_set)
    if normalized_choices and normalized not in normalized_choices:
        return ParsedMedicalAnswer(
            answer=boxed_raw,
            boxed_raw=boxed_raw,
            normalized_choice=normalized,
            parse_status="choice_out_of_set",
            policy=policy,
            boxed_count=boxed_count,
            used_answer_block=used_answer_block,
            is_abstain=False,
            is_valid_choice=False,
        )

    return ParsedMedicalAnswer(
        answer=boxed_raw,
        boxed_raw=boxed_raw,
        normalized_choice=normalized,
        parse_status="ok_choice",
        policy=policy,
        boxed_count=boxed_count,
        used_answer_block=used_answer_block,
        is_abstain=False,
        is_valid_choice=True,
    )


__all__ = [
    "POLICY_FINAL_ANSWER_LAST_BOXED",
    "POLICY_FIRST_BOXED_LEGACY",
    "POLICY_SINGLE_BOXED_STRICT",
    "ParsedMedicalAnswer",
    "extract_single_boxed",
    "find_boxed_answers",
    "is_abstain_boxed",
    "normalize_choice",
    "parse_medical_answer",
]
