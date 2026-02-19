#!/usr/bin/env python3
"""Merge LoRA adapter into base model for Stage C."""

from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint into base model.")
    parser.add_argument("--base_model_path", required=True, type=str)
    parser.add_argument("--lora_model_path", required=True, type=str)
    parser.add_argument("--merged_model_save_path", required=True, type=str)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=args.trust_remote_code)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )

    lora_model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(args.merged_model_save_path, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_model_save_path)

    print("[PASS] LoRA merged.")
    print(f"  base_model_path={args.base_model_path}")
    print(f"  lora_model_path={args.lora_model_path}")
    print(f"  merged_model_save_path={args.merged_model_save_path}")


if __name__ == "__main__":
    main()
