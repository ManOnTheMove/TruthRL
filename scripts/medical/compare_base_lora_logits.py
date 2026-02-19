#!/usr/bin/env python3
"""Compare next-token logits between base model and base+LoRA runtime model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare logits base vs base+LoRA.")
    parser.add_argument("--base-model-dir", required=True, type=Path)
    parser.add_argument("--lora-adapter-dir", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def top_tokens(logits: torch.Tensor, tokenizer, k: int):
    vals, ids = torch.topk(logits, k=k, dim=-1)
    return [
        {"id": int(i.item()), "token": tokenizer.convert_ids_to_tokens([int(i.item())])[0], "logit": float(v.item())}
        for v, i in zip(vals, ids)
    ]


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    prompt = pq.read_table(args.parquet, columns=["prompt"]).slice(args.index, 1).column("prompt").to_pylist()[0]

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)

    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_dir),
        trust_remote_code=True,
        torch_dtype=model_dtype,
        device_map="auto",
    )
    base.eval()
    in_base = {k: v.to(base.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits_base = base(**in_base).logits[0, -1, :].float().cpu()
    del base
    torch.cuda.empty_cache()

    lora_base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_dir),
        trust_remote_code=True,
        torch_dtype=model_dtype,
        device_map="auto",
    )
    lora_model = PeftModel.from_pretrained(lora_base, str(args.lora_adapter_dir))
    lora_model = lora_model.merge_and_unload()
    lora_model.eval()
    in_lora = {k: v.to(lora_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits_lora = lora_model(**in_lora).logits[0, -1, :].float().cpu()

    diff = logits_lora - logits_base
    out = {
        "index": args.index,
        "dtype": args.dtype,
        "prompt_tail": prompt[-220:],
        "mean_abs_diff": float(torch.mean(torch.abs(diff)).item()),
        "max_abs_diff": float(torch.max(torch.abs(diff)).item()),
        "argmax_base": int(torch.argmax(logits_base).item()),
        "argmax_lora": int(torch.argmax(logits_lora).item()),
        "argmax_same": int(torch.argmax(logits_base).item()) == int(torch.argmax(logits_lora).item()),
        "topk_base": top_tokens(logits_base, tokenizer, args.topk),
        "topk_lora": top_tokens(logits_lora, tokenizer, args.topk),
    }
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print(
        json.dumps(
            {
                "mean_abs_diff": out["mean_abs_diff"],
                "max_abs_diff": out["max_abs_diff"],
                "argmax_same": out["argmax_same"],
                "argmax_base": out["argmax_base"],
                "argmax_lora": out["argmax_lora"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
