"""Prompt templates for medical data pipeline."""

from __future__ import annotations

TEMPLATE_VERSION = "medical_prompt_v2_d26_idk"

MEDICAL_RL_PROMPT_TEMPLATE = """You are a medical expert with advanced clinical reasoning skills.
Answer the following multiple-choice question.

Your response MUST include exactly two sections:
1) A reasoning section enclosed by <think> and </think>.
2) A final answer section enclosed by <answer> and </answer>.

The final boxed action MUST be exactly one of:
- Option answer: \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, or \\boxed{{E}}
- Abstention when knowledge is insufficient: \\boxed{{I don't know}}

Do not output any boxed content other than the final boxed action.

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

The final boxed action MUST be exactly one of:
- Option answer: \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, or \\boxed{{E}}
- Abstention when knowledge is insufficient: \\boxed{{I don't know}}

Do not output any boxed content other than the final boxed action.

Question:
{question}
"""
