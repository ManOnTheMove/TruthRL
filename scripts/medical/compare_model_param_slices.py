#!/usr/bin/env python3
"""Compare parameter slices between two HF model directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare selected parameter slices between two models.")
    parser.add_argument("--base-model-dir", required=True, type=Path)
    parser.add_argument("--other-model-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--slice-size", type=int, default=2048)
    return parser.parse_args()


def collect_slices(model: torch.nn.Module, slice_size: int) -> dict[str, list[float]]:
    wanted = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.10.self_attn.k_proj.weight",
        "model.layers.20.mlp.down_proj.weight",
        "lm_head.weight",
    ]
    out: dict[str, list[float]] = {}
    sd = model.state_dict()
    for name in wanted:
        if name not in sd:
            continue
        t = sd[name].detach().float().view(-1)[:slice_size].cpu()
        out[name] = t.tolist()
    return out


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_dir), trust_remote_code=True, device_map="cpu", torch_dtype="auto"
    )
    base_slices = collect_slices(base, args.slice_size)
    del base
    torch.cuda.empty_cache()

    other = AutoModelForCausalLM.from_pretrained(
        str(args.other_model_dir), trust_remote_code=True, device_map="cpu", torch_dtype="auto"
    )
    other_slices = collect_slices(other, args.slice_size)
    del other
    torch.cuda.empty_cache()

    compare = {}
    all_equal = True
    for k, v in base_slices.items():
        ov = other_slices.get(k)
        if ov is None:
            compare[k] = {"present_in_other": False}
            all_equal = False
            continue
        t1 = torch.tensor(v)
        t2 = torch.tensor(ov)
        eq = torch.equal(t1, t2)
        mad = float(torch.mean(torch.abs(t1 - t2)).item())
        mxd = float(torch.max(torch.abs(t1 - t2)).item())
        compare[k] = {"equal": bool(eq), "mean_abs_diff": mad, "max_abs_diff": mxd, "n": int(t1.numel())}
        all_equal = all_equal and eq

    out = {
        "base_model_dir": str(args.base_model_dir),
        "other_model_dir": str(args.other_model_dir),
        "slice_size": args.slice_size,
        "all_equal": all_equal,
        "compare": compare,
    }
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] wrote {args.out_json}")
    print(json.dumps({"all_equal": all_equal, "checked_params": list(compare.keys())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
