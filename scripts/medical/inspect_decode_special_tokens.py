#!/usr/bin/env python3
"""Inspect model decode outputs with/without skip_special_tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect decode behavior wrt special tokens.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet, columns=["prompt"]).slice(args.index, 1)
    prompts = table.column("prompt").to_pylist()
    if not prompts:
        raise ValueError(f"index {args.index} out of range")
    prompt = prompts[0]

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

    decoded_skip_true = tokenizer.decode(gen_ids, skip_special_tokens=True)
    decoded_skip_false = tokenizer.decode(gen_ids, skip_special_tokens=False)
    gen_tokens = tokenizer.convert_ids_to_tokens(gen_ids.tolist())

    out = {
        "index": args.index,
        "model_dir": str(args.model_dir),
        "decoded_skip_special_tokens_true": decoded_skip_true,
        "decoded_skip_special_tokens_false": decoded_skip_false,
        "first_80_tokens": gen_tokens[:80],
    }
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print("head_skip_true:", decoded_skip_true[:200].replace("\n", " "))
    print("head_skip_false:", decoded_skip_false[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
