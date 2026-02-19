#!/usr/bin/env python3
"""Estimate effective LoRA delta magnitude from adapter tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate LoRA delta magnitude.")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--lora-alpha", type=float, required=True)
    parser.add_argument("--lora-r", type=float, required=True)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--sample-max-modules", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(args.adapter_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors in {args.adapter_dir}")

    state = {}
    for f in files:
        state.update(load_file(str(f)))

    scale = args.lora_alpha / args.lora_r
    prefixes = set()
    for k in state:
        if ".lora_A.weight" in k:
            prefixes.add(k.replace(".lora_A.weight", ""))

    rows = []
    for i, pfx in enumerate(sorted(prefixes)):
        if i >= args.sample_max_modules:
            break
        ka = f"{pfx}.lora_A.weight"
        kb = f"{pfx}.lora_B.weight"
        if ka not in state or kb not in state:
            continue
        a = state[ka].float()
        b = state[kb].float()
        delta = (b @ a) * scale
        rows.append(
            {
                "module": pfx,
                "delta_mean_abs": float(delta.abs().mean().item()),
                "delta_max_abs": float(delta.abs().max().item()),
                "a_mean_abs": float(a.abs().mean().item()),
                "b_mean_abs": float(b.abs().mean().item()),
                "shape_A": list(a.shape),
                "shape_B": list(b.shape),
            }
        )

    rows.sort(key=lambda x: x["delta_mean_abs"], reverse=True)
    out = {
        "adapter_dir": str(args.adapter_dir),
        "scale": scale,
        "num_modules_sampled": len(rows),
        "global_delta_mean_abs": (sum(r["delta_mean_abs"] for r in rows) / len(rows)) if rows else 0.0,
        "global_delta_max_abs": max((r["delta_max_abs"] for r in rows), default=0.0),
        "top5": rows[:5],
        "bottom5": rows[-5:] if len(rows) >= 5 else rows,
    }
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print(
        json.dumps(
            {
                "num_modules_sampled": out["num_modules_sampled"],
                "global_delta_mean_abs": out["global_delta_mean_abs"],
                "global_delta_max_abs": out["global_delta_max_abs"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
