#!/usr/bin/env python3
"""Merge LoRA adapter into base model for Stage C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_and_validate_adapter_config(lora_model_path: str, expected_lora_alpha: float | None = None) -> dict:
    adapter_config = Path(lora_model_path) / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(f"adapter_config.json not found: {adapter_config}")

    data = json.loads(adapter_config.read_text(encoding="utf-8"))
    if "lora_alpha" not in data:
        raise ValueError(f"adapter config missing lora_alpha: {adapter_config}")

    lora_alpha = float(data["lora_alpha"])
    if lora_alpha <= 0:
        raise ValueError(f"adapter config lora_alpha must be > 0, got {data['lora_alpha']}: {adapter_config}")
    if expected_lora_alpha is not None and abs(lora_alpha - expected_lora_alpha) > 1e-9:
        raise ValueError(
            f"adapter config lora_alpha mismatch: expected {expected_lora_alpha}, got {data['lora_alpha']} "
            f"in {adapter_config}"
        )

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint into base model.")
    parser.add_argument("--base_model_path", required=True, type=str)
    parser.add_argument("--lora_model_path", required=True, type=str)
    parser.add_argument("--merged_model_save_path", required=True, type=str)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-lora-alpha", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adapter_config = _load_and_validate_adapter_config(args.lora_model_path, args.expected_lora_alpha)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    print(f"  lora_alpha={adapter_config['lora_alpha']}")
    print(f"  merged_model_save_path={args.merged_model_save_path}")


if __name__ == "__main__":
    main()
