#!/usr/bin/env python3
"""Collect full inference examples from a merged model for reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect selected inference examples.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        required=True,
        help="0-based row indices in the parquet prompt column.",
    )
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    max_idx = max(args.indices)
    table = pq.read_table(args.parquet, columns=["prompt"]).slice(0, max_idx + 1)
    prompts = table.column("prompt").to_pylist()

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
    for i in args.indices:
        prompt = prompts[i]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        records.append(
            {
                "index": i,
                "prompt": prompt,
                "output": text,
                "has_answer_tag": "<answer>" in text and "</answer>" in text,
                "has_think_tag": "<think>" in text and "</think>" in text,
                "has_boxed": "\\boxed{" in text or "boxed{" in text,
            }
        )

    args.out_json.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    for r in records:
        print(
            json.dumps(
                {
                    "index": r["index"],
                    "has_answer_tag": r["has_answer_tag"],
                    "has_think_tag": r["has_think_tag"],
                    "has_boxed": r["has_boxed"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
