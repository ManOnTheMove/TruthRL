"""Prompt templates frozen in Stage B."""

from __future__ import annotations

TEMPLATE_VERSION = "medical_prompt_v1"

MEDICAL_RL_PROMPT_TEMPLATE = """You are a medical expert with advanced clinical reasoning skills.
Answer the following multiple-choice question.

Your response MUST include exactly two sections:
1) A reasoning section enclosed by <think> and </think>.
2) A final answer section enclosed by <answer> and </answer>.

The final selected option MUST appear in \\boxed{{}} (for example, \\boxed{{A}}).

Question:
{question}
"""

MEDICAL_SFT_PROMPT_TEMPLATE = """You are a medical expert with advanced clinical reasoning skills.
Solve the following multiple-choice question.

Your response format must be:
<think>
...
</think>
<answer>
...
</answer>

The final selected option MUST appear in \\boxed{{}}.

Question:
{question}
"""

