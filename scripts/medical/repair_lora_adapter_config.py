#!/usr/bin/env python3
"""Repair LoRA adapter_config.json generated from FSDP fallback merge.

Some FSDP merge flows can emit adapter_config.json with lora_alpha=0, which
makes LoRA scaling exactly zero at inference. This utility enforces a positive
alpha (default: 16) to keep adapter effect non-zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair LoRA adapter config values.")
    parser.add_argument("--adapter-config", required=True, type=Path)
    parser.add_argument("--expected-lora-alpha", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.adapter_config.is_file():
        raise FileNotFoundError(f"adapter config not found: {args.adapter_config}")
    if args.expected_lora_alpha <= 0:
        raise ValueError("--expected-lora-alpha must be > 0")

    data = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    before_alpha = data.get("lora_alpha", None)
    repaired = False

    if before_alpha is None or float(before_alpha) <= 0:
        data["lora_alpha"] = float(args.expected_lora_alpha)
        repaired = True

    args.adapter_config.write_text(
        json.dumps(data, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "adapter_config": str(args.adapter_config),
                "before_lora_alpha": before_alpha,
                "after_lora_alpha": data.get("lora_alpha"),
                "repaired": repaired,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
