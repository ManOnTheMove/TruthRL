#!/usr/bin/env python3
"""Comprehensive Stage-C retest runner (A/B/C/D) for g17.

This script executes:
1) Format hit vs decode window (A)
2) Boxed parseability/accuracy on GRPO eval set (B)
3) Distill truncation quantification (C)
4) Chat-template sensitivity check (D)
and produces a final retrain gate decision.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-Za-z])\s*\}")


@dataclass
class GenStats:
    complete_think_tag: int = 0
    complete_answer_tag: int = 0
    boxed: int = 0
    start_close_answer: int = 0
    start_close_think: int = 0
    trunc_hit: int = 0
    generated_tokens_sum: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-C comprehensive retest.")
    parser.add_argument(
        "--sft_val_parquet",
        type=Path,
        default=Path("data/medical/verl/medqa_sft_val.parquet"),
    )
    parser.add_argument(
        "--grpo_test_parquet",
        type=Path,
        default=Path("data/medical/verl/medqa_grpo_test.parquet"),
    )
    parser.add_argument(
        "--distill_dir",
        type=Path,
        default=Path("data/medical/distill"),
    )
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=Path("/home/erichyu/projects/def-zshakeri/erichyu/embc/models/Qwen3-8B"),
    )
    parser.add_argument(
        "--fix_model_path",
        type=Path,
        default=Path("/home/erichyu/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/stagec_c8_8b_full_fixalpha/merged_model"),
    )
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--windows", type=str, default="512,1024,1536,2048")
    parser.add_argument("--template_window", type=int, default=1536)
    parser.add_argument("--max_new_tokens_b", type=int, default=2048)
    parser.add_argument("--truncation_high_threshold", type=float, default=0.5)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    return parser.parse_args()


def _ensure_paths(args: argparse.Namespace) -> None:
    required = [
        args.sft_val_parquet,
        args.grpo_test_parquet,
        args.distill_dir,
        args.base_model_path / "config.json",
        args.fix_model_path / "config.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")
    if args.sample_size < 50:
        raise ValueError(f"sample_size must be >= 50, got {args.sample_size}")


def _parse_windows(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("windows must be non-empty")
    return vals


def _extract_boxed_choice(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def _format_prompt(prompt_obj: Any, tokenizer, apply_chat_template: bool) -> str:
    if isinstance(prompt_obj, list):
        messages = prompt_obj
        if apply_chat_template:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        chunks = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            chunks.append(f"{role}: {content}")
        return "\n".join(chunks).strip()

    prompt_text = str(prompt_obj)
    if apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt_text


def _load_sft_prompts(path: Path, n: int) -> list[Any]:
    table = pq.read_table(path, columns=["prompt"]).slice(0, n)
    return table.column("prompt").to_pylist()


def _load_grpo_items(path: Path, n: int) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["prompt", "reward_model"]).slice(0, n)
    prompts = table.column("prompt").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    items: list[dict[str, Any]] = []
    for p, rm in zip(prompts, reward_models):
        gt = rm.get("ground_truth", {}) if isinstance(rm, dict) else {}
        target = gt.get("target", [])
        target_letter = None
        if isinstance(target, list) and target:
            target_letter = str(target[0]).strip().upper()
        items.append({"prompt": p, "target": target_letter})
    return items


def _run_generation_metrics(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
) -> tuple[GenStats, list[str], list[int]]:
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    stats = GenStats()
    texts: list[str] = []
    generated_tokens: list[int] = []

    for out in outputs:
        first = out.outputs[0]
        text = first.text
        tok_len = len(first.token_ids) if first.token_ids is not None else 0
        texts.append(text)
        generated_tokens.append(tok_len)

        has_open_think = "<think>" in text
        has_close_think = "</think>" in text
        has_open_answer = "<answer>" in text
        has_close_answer = "</answer>" in text
        has_boxed = _extract_boxed_choice(text) is not None
        if has_open_think and has_close_think:
            stats.complete_think_tag += 1
        if has_open_answer and has_close_answer:
            stats.complete_answer_tag += 1
        if has_boxed:
            stats.boxed += 1
        if text.lstrip().startswith("</answer>"):
            stats.start_close_answer += 1
        if text.lstrip().startswith("</think>"):
            stats.start_close_think += 1
        if tok_len >= sampling_params.max_tokens:
            stats.trunc_hit += 1
        stats.generated_tokens_sum += tok_len

    return stats, texts, generated_tokens


def _stats_to_rate_dict(stats: GenStats, n: int) -> dict[str, Any]:
    return {
        "complete_think_tag_rate": stats.complete_think_tag / n,
        "complete_answer_tag_rate": stats.complete_answer_tag / n,
        "boxed_rate": stats.boxed / n,
        "start_close_answer_rate": stats.start_close_answer / n,
        "start_close_think_rate": stats.start_close_think / n,
        "truncation_hit_rate": stats.trunc_hit / n,
        "avg_generated_tokens": stats.generated_tokens_sum / n,
        "raw_counts": {
            "complete_think_tag": stats.complete_think_tag,
            "complete_answer_tag": stats.complete_answer_tag,
            "boxed": stats.boxed,
            "start_close_answer": stats.start_close_answer,
            "start_close_think": stats.start_close_think,
            "trunc_hit": stats.trunc_hit,
            "generated_tokens_sum": stats.generated_tokens_sum,
            "sample_size": n,
        },
    }


def _run_model_suite(
    model_path: Path,
    sft_prompts: list[Any],
    grpo_items: list[dict[str, Any]],
    windows: list[int],
    template_window: int,
    max_new_tokens_b: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
    )

    sft_prompts_chat = [_format_prompt(x, tokenizer, True) for x in sft_prompts]
    sft_prompts_raw = [_format_prompt(x, tokenizer, False) for x in sft_prompts]
    grpo_prompts_chat = [_format_prompt(x["prompt"], tokenizer, True) for x in grpo_items]
    grpo_targets = [x["target"] for x in grpo_items]

    # A: apply_chat_template=True, multi-window
    a_results: dict[str, Any] = {}
    for w in windows:
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=w, n=1, skip_special_tokens=True)
        stats, _, _ = _run_generation_metrics(llm, params, sft_prompts_chat)
        a_results[str(w)] = _stats_to_rate_dict(stats, len(sft_prompts_chat))

    # D: template sensitivity at one fixed window
    params_d = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=template_window,
        n=1,
        skip_special_tokens=True,
    )
    stats_true, _, _ = _run_generation_metrics(llm, params_d, sft_prompts_chat)
    stats_false, _, _ = _run_generation_metrics(llm, params_d, sft_prompts_raw)
    d_results = {
        "apply_chat_template_true": _stats_to_rate_dict(stats_true, len(sft_prompts_chat)),
        "apply_chat_template_false": _stats_to_rate_dict(stats_false, len(sft_prompts_raw)),
    }

    # B: boxed parseability and accuracy
    params_b = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens_b,
        n=1,
        skip_special_tokens=True,
    )
    stats_b, texts_b, generated_b = _run_generation_metrics(llm, params_b, grpo_prompts_chat)
    extract_success = 0
    option_valid = 0
    accuracy = 0
    accuracy_on_extracted = 0
    for text, target in zip(texts_b, grpo_targets):
        pred = _extract_boxed_choice(text)
        if pred is not None:
            extract_success += 1
            if "A" <= pred <= "Z":
                option_valid += 1
            if target is not None and pred == target:
                accuracy += 1
                accuracy_on_extracted += 1

    n_b = len(grpo_items)
    b_results = {
        "sample_size": n_b,
        "boxed_extract_success_rate": extract_success / n_b,
        "boxed_option_valid_rate": option_valid / n_b,
        "boxed_accuracy": accuracy / n_b,
        "boxed_accuracy_on_extracted": (accuracy_on_extracted / extract_success) if extract_success > 0 else 0.0,
        "truncation_hit_rate": stats_b.trunc_hit / n_b,
        "avg_generated_tokens": statistics.mean(generated_b) if generated_b else 0.0,
        "raw_counts": {
            "extract_success": extract_success,
            "option_valid": option_valid,
            "accuracy": accuracy,
            "sample_size": n_b,
            "trunc_hit": stats_b.trunc_hit,
        },
    }

    del llm

    return {
        "a_format_vs_window": a_results,
        "b_parseability_accuracy": b_results,
        "d_template_sensitivity": d_results,
    }


def _token_pos(text: str, marker: str, tokenizer) -> int | None:
    idx = text.find(marker)
    if idx < 0:
        return None
    prefix = text[:idx]
    return len(tokenizer(prefix, add_special_tokens=False)["input_ids"])


def _run_c_distill_stats(distill_dir: Path, tokenizer) -> dict[str, Any]:
    files = sorted(distill_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No distill files found in {distill_dir}")

    total_lens: list[int] = []
    close_think_pos: list[int | None] = []
    open_answer_pos: list[int | None] = []
    boxed_pos: list[int | None] = []

    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        total_lens.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        close_think_pos.append(_token_pos(text, "</think>", tokenizer))
        open_answer_pos.append(_token_pos(text, "<answer>", tokenizer))
        boxed_pos.append(_token_pos(text, "\\boxed{", tokenizer))

    n = len(files)
    frac_total_len_gt_512 = sum(x > 512 for x in total_lens) / n
    frac_close_think_after_512 = sum((x is not None and x > 512) for x in close_think_pos) / n
    frac_open_answer_after_512 = sum((x is not None and x > 512) for x in open_answer_pos) / n
    frac_boxed_after_512 = sum((x is not None and x > 512) for x in boxed_pos) / n

    sorted_lens = sorted(total_lens)
    p50 = sorted_lens[n // 2]
    p90 = sorted_lens[int(n * 0.9)]
    p95 = sorted_lens[int(n * 0.95)]

    return {
        "n_files": n,
        "token_length": {
            "mean": statistics.mean(total_lens),
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "max": max(total_lens),
        },
        "frac_total_len_gt_512": frac_total_len_gt_512,
        "frac_close_think_after_512": frac_close_think_after_512,
        "frac_open_answer_after_512": frac_open_answer_after_512,
        "frac_boxed_after_512": frac_boxed_after_512,
    }


def _decision_gate(
    base: dict[str, Any],
    fix: dict[str, Any],
    truncation_high_threshold: float,
) -> dict[str, Any]:
    base_a = base["a_format_vs_window"]
    fix_a = fix["a_format_vs_window"]
    base_b = base["b_parseability_accuracy"]
    fix_b = fix["b_parseability_accuracy"]

    cond1 = (fix_a["1536"]["boxed_rate"] <= base_a["1536"]["boxed_rate"]) or (
        fix_a["2048"]["boxed_rate"] <= base_a["2048"]["boxed_rate"]
    )
    cond2 = (fix_a["1536"]["truncation_hit_rate"] >= truncation_high_threshold) or (
        fix_a["2048"]["truncation_hit_rate"] >= truncation_high_threshold
    )
    cond3 = fix_b["boxed_accuracy"] <= base_b["boxed_accuracy"]

    retrain_now = cond1 or cond2 or cond3
    hold_retrain = (not cond1) and (not cond2) and (not cond3)

    return {
        "trigger_retrain_conditions": {
            "fixalpha_boxed_rate_not_better_than_base_on_1536_or_2048": cond1,
            "long_window_truncation_hit_rate_high": cond2,
            "boxed_accuracy_no_improvement_or_drop": cond3,
        },
        "retrain_now": retrain_now,
        "hold_retrain_only_if_all_conditions_false": hold_retrain,
        "truncation_high_threshold": truncation_high_threshold,
    }


def _template_policy(base_d: dict[str, Any], fix_d: dict[str, Any]) -> dict[str, Any]:
    # "Huge difference" criterion for policy lock.
    def _gap(d: dict[str, Any], key: str) -> float:
        t = d["apply_chat_template_true"][key]
        f = d["apply_chat_template_false"][key]
        return abs(t - f)

    gaps = {
        "base_boxed_rate_gap": _gap(base_d, "boxed_rate"),
        "base_answer_tag_gap": _gap(base_d, "complete_answer_tag_rate"),
        "fix_boxed_rate_gap": _gap(fix_d, "boxed_rate"),
        "fix_answer_tag_gap": _gap(fix_d, "complete_answer_tag_rate"),
    }
    huge = any(v >= 0.15 for v in gaps.values())
    return {
        "gaps": gaps,
        "huge_difference": huge,
        "evaluation_protocol_fixed_apply_chat_template_true": True if huge else True,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    base = report["model_results"]["base"]
    fix = report["model_results"]["fixalpha"]
    gate = report["decision_gate"]
    c_stats = report["c_distill_truncation"]
    tpl = report["template_policy"]

    lines: list[str] = []
    lines.append("# Stage C 完整复测报告（g17）")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Sample size: `{report['config']['sample_size']}`")
    lines.append(f"- Windows (A): `{report['config']['windows']}`")
    lines.append("")
    lines.append("## A: 格式命中 vs 解码窗口")
    lines.append("")
    lines.append("| Window | base boxed | fix boxed | base trunc | fix trunc |")
    lines.append("|---|---:|---:|---:|---:|")
    for w in report["config"]["windows"]:
        ws = str(w)
        lines.append(
            f"| {ws} | {base['a_format_vs_window'][ws]['boxed_rate']:.3f} | "
            f"{fix['a_format_vs_window'][ws]['boxed_rate']:.3f} | "
            f"{base['a_format_vs_window'][ws]['truncation_hit_rate']:.3f} | "
            f"{fix['a_format_vs_window'][ws]['truncation_hit_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## B: 可解析性与准确性（2048）")
    lines.append("")
    lines.append("| Model | boxed_extract_success | boxed_valid | boxed_accuracy |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| base | {base['b_parseability_accuracy']['boxed_extract_success_rate']:.3f} | "
        f"{base['b_parseability_accuracy']['boxed_option_valid_rate']:.3f} | "
        f"{base['b_parseability_accuracy']['boxed_accuracy']:.3f} |"
    )
    lines.append(
        f"| fixalpha | {fix['b_parseability_accuracy']['boxed_extract_success_rate']:.3f} | "
        f"{fix['b_parseability_accuracy']['boxed_option_valid_rate']:.3f} | "
        f"{fix['b_parseability_accuracy']['boxed_accuracy']:.3f} |"
    )
    lines.append("")
    lines.append("## C: 训练截断定量复核（distill）")
    lines.append("")
    lines.append(f"- `frac_total_len_gt_512`: `{c_stats['frac_total_len_gt_512']:.4f}`")
    lines.append(f"- `frac_close_think_after_512`: `{c_stats['frac_close_think_after_512']:.4f}`")
    lines.append(f"- `frac_open_answer_after_512`: `{c_stats['frac_open_answer_after_512']:.4f}`")
    lines.append(f"- `frac_boxed_after_512`: `{c_stats['frac_boxed_after_512']:.4f}`")
    lines.append("")
    lines.append("## D: 模板敏感性")
    lines.append("")
    lines.append(f"- Huge difference: `{tpl['huge_difference']}`")
    lines.append(f"- Fixed protocol (`apply_chat_template=True`): `{tpl['evaluation_protocol_fixed_apply_chat_template_true']}`")
    lines.append(f"- Gaps: `{json.dumps(tpl['gaps'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## 门禁结论")
    lines.append("")
    lines.append(f"- `retrain_now`: `{gate['retrain_now']}`")
    lines.append(f"- Trigger details: `{json.dumps(gate['trigger_retrain_conditions'], ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    _ensure_paths(args)
    windows = _parse_windows(args.windows)

    sft_prompts = _load_sft_prompts(args.sft_val_parquet, args.sample_size)
    grpo_items = _load_grpo_items(args.grpo_test_parquet, args.sample_size)

    base_results = _run_model_suite(
        args.base_model_path,
        sft_prompts=sft_prompts,
        grpo_items=grpo_items,
        windows=windows,
        template_window=args.template_window,
        max_new_tokens_b=args.max_new_tokens_b,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    fix_results = _run_model_suite(
        args.fix_model_path,
        sft_prompts=sft_prompts,
        grpo_items=grpo_items,
        windows=windows,
        template_window=args.template_window,
        max_new_tokens_b=args.max_new_tokens_b,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    tokenizer_for_c = AutoTokenizer.from_pretrained(str(args.base_model_path), trust_remote_code=True)
    c_stats = _run_c_distill_stats(args.distill_dir, tokenizer_for_c)
    gate = _decision_gate(base_results, fix_results, args.truncation_high_threshold)
    template_policy = _template_policy(
        base_results["d_template_sensitivity"],
        fix_results["d_template_sensitivity"],
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sample_size": args.sample_size,
            "windows": windows,
            "template_window": args.template_window,
            "max_new_tokens_b": args.max_new_tokens_b,
            "truncation_high_threshold": args.truncation_high_threshold,
            "sft_val_parquet": str(args.sft_val_parquet),
            "grpo_test_parquet": str(args.grpo_test_parquet),
            "distill_dir": str(args.distill_dir),
            "base_model_path": str(args.base_model_path),
            "fix_model_path": str(args.fix_model_path),
        },
        "model_results": {
            "base": base_results,
            "fixalpha": fix_results,
        },
        "c_distill_truncation": c_stats,
        "template_policy": template_policy,
        "decision_gate": gate,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(f"[PASS] wrote json: {args.out_json}")
    print(f"[PASS] wrote md:   {args.out_md}")
    print(json.dumps({"retrain_now": gate["retrain_now"], "trigger": gate["trigger_retrain_conditions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
