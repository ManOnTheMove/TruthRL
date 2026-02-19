#!/usr/bin/env python3
"""Run a small output-quality audit for a merged Stage C model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit merged model outputs on a parquet prompt set.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet, columns=["prompt"]).slice(0, args.num_samples)
    prompts = table.column("prompt").to_pylist()
    if not prompts:
        raise ValueError(f"No prompts found in {args.parquet}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    records = []
    boxed_count = 0
    answer_tag_count = 0
    think_tag_count = 0
    total_generated_tokens = 0

    for idx, prompt in enumerate(prompts, start=1):
        if args.apply_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt_text = prompt

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_input_length).to(
            model.device
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
        num_generated_tokens = int(gen_ids.shape[0])
        total_generated_tokens += num_generated_tokens
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        has_boxed = "\\boxed{" in text or "boxed{" in text
        has_answer = "<answer>" in text and "</answer>" in text
        has_think = "<think>" in text and "</think>" in text
        boxed_count += int(has_boxed)
        answer_tag_count += int(has_answer)
        think_tag_count += int(has_think)

        records.append(
            {
                "idx": idx,
                "prompt_preview": prompt[:240],
                "output_preview": text[:500],
                "num_generated_tokens": num_generated_tokens,
                "has_boxed": has_boxed,
                "has_answer_tag": has_answer,
                "has_think_tag": has_think,
            }
        )

    sample_size = len(prompts)
    summary = {
        "sample_size": sample_size,
        "boxed_hit": boxed_count,
        "answer_tag_hit": answer_tag_count,
        "think_tag_hit": think_tag_count,
        "boxed_hit_rate": boxed_count / sample_size,
        "answer_tag_hit_rate": answer_tag_count / sample_size,
        "think_tag_hit_rate": think_tag_count / sample_size,
        "avg_generated_tokens": total_generated_tokens / sample_size,
        "apply_chat_template": bool(args.apply_chat_template),
        "records": records,
    }
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print(
        json.dumps(
            {
                "sample_size": sample_size,
                "boxed_hit": boxed_count,
                "answer_tag_hit": answer_tag_count,
                "think_tag_hit": think_tag_count,
                "boxed_hit_rate": summary["boxed_hit_rate"],
                "answer_tag_hit_rate": summary["answer_tag_hit_rate"],
                "think_tag_hit_rate": summary["think_tag_hit_rate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
