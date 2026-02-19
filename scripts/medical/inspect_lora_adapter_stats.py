#!/usr/bin/env python3
"""Inspect magnitude statistics of LoRA adapter tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LoRA adapter tensor stats.")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(args.adapter_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {args.adapter_dir}")

    state = {}
    for f in files:
        state.update(load_file(str(f)))

    stats = []
    for k, v in state.items():
        if "lora_" not in k:
            continue
        stats.append(
            {
                "name": k,
                "mean_abs": float(v.abs().mean().item()),
                "max_abs": float(v.abs().max().item()),
                "numel": int(v.numel()),
            }
        )
    stats.sort(key=lambda x: x["mean_abs"], reverse=True)

    out = {
        "adapter_dir": str(args.adapter_dir),
        "n_files": len(files),
        "files": [f.name for f in files],
        "n_tensors_total": len(state),
        "n_lora_tensors": len(stats),
        "global_mean_abs": (sum(x["mean_abs"] for x in stats) / len(stats)) if stats else 0.0,
        "global_max_abs": max((x["max_abs"] for x in stats), default=0.0),
        "top5": stats[:5],
        "bottom5": stats[-5:] if len(stats) >= 5 else stats,
    }
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print(
        json.dumps(
            {
                "n_lora_tensors": out["n_lora_tensors"],
                "global_mean_abs": out["global_mean_abs"],
                "global_max_abs": out["global_max_abs"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
