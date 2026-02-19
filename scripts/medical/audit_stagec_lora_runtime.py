#!/usr/bin/env python3
"""Audit outputs by loading base model + external LoRA adapter at runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit base+LoRA runtime model outputs.")
    parser.add_argument("--base-model-dir", required=True, type=Path)
    parser.add_argument("--lora-adapter-dir", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet, columns=["prompt"]).slice(0, args.num_samples)
    prompts = table.column("prompt").to_pylist()
    if not prompts:
        raise ValueError("No prompts found.")

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.lora_adapter_dir))
    model = model.merge_and_unload()
    model.eval()

    records = []
    boxed_count = 0
    answer_tag_count = 0
    think_tag_count = 0

    for idx, prompt in enumerate(prompts, start=1):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length).to(model.device)
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
                "has_boxed": has_boxed,
                "has_answer_tag": has_answer,
                "has_think_tag": has_think,
            }
        )

    n = len(prompts)
    summary = {
        "sample_size": n,
        "boxed_hit": boxed_count,
        "answer_tag_hit": answer_tag_count,
        "think_tag_hit": think_tag_count,
        "boxed_hit_rate": boxed_count / n,
        "answer_tag_hit_rate": answer_tag_count / n,
        "think_tag_hit_rate": think_tag_count / n,
        "records": records,
    }
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print(
        json.dumps(
            {
                "sample_size": n,
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
