#!/usr/bin/env python3
"""Stage D2.5 KBP collection for MedQA GRPO train split.

This script samples 256 responses per question (chunked), writes raw responses
and parsed records to parquet(zstd) shards, then builds question-level KBP
labels and quality reports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoTokenizer


MONITOR_REPORT_INTERVAL_S = 600.0
MONITOR_SAMPLE_INTERVAL_S = 30.0


@dataclass
class QuestionRecord:
    question_id: str
    question_rank: int
    split: str
    prompt_id: str
    prompt_obj: Any
    prompt_text: str
    target_set: set[str]
    choice_set: set[str]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Collect KBP responses for MedQA.")
    parser.add_argument(
        "--input_parquet",
        type=Path,
        default=repo_root / "data" / "medical" / "verl" / "medqa_grpo_train.parquet",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=repo_root / "data" / "medical" / "kbp",
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--split", type=str, default="medqa_grpo_train")
    parser.add_argument("--question_limit", type=int, default=0)
    parser.add_argument("--question_rank_start", type=int, default=0)
    parser.add_argument("--question_rank_end", type=int, default=-1)
    parser.add_argument("--probes_per_question", type=int, default=256)
    parser.add_argument("--n_chunk", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(
            "/home/erichyu/scratch/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model"
        ),
    )
    parser.add_argument("--model_id", type=str, default="stagec-c8-8b-merged")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=10240)
    parser.add_argument("--stop_str", type=str, default="</answer>")
    parser.add_argument("--include_stop_str_in_output", action="store_true", default=True)
    parser.add_argument("--seed_base", type=int, default=20260301)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--part_rows", type=int, default=80000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess_only", action="store_true")
    parser.add_argument("--ook_rule_version", type=str, default="kbp_v1_correct_eq_zero_probe256")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_reward_module(repo_root: Path):
    reward_path = repo_root / "training" / "verl" / "verl" / "utils" / "reward_score" / "clinical_medqa_reward.py"
    if not reward_path.exists():
        raise FileNotFoundError(f"reward module not found: {reward_path}")
    spec = importlib.util.spec_from_file_location("clinical_medqa_reward", str(reward_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward module from {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_target_set(module: Any, ground_truth: dict[str, Any]) -> set[str]:
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    out: set[str] = set()
    if isinstance(targets, list):
        for t in targets:
            n = module.normalize_choice(str(t))
            if n:
                out.add(n)
    return out


def _extract_choice_set(module: Any, ground_truth: dict[str, Any]) -> set[str]:
    choices = ground_truth.get("choices", [])
    out: set[str] = set()
    if isinstance(choices, list):
        for c in choices:
            if not isinstance(c, str):
                continue
            n = module.normalize_choice(c)
            if n:
                out.add(n)
                continue
            if ":" in c:
                n2 = module.normalize_choice(c.split(":", 1)[0])
                if n2:
                    out.add(n2)
    return out


def _format_prompt(tokenizer: Any, prompt_obj: Any) -> str:
    if isinstance(prompt_obj, list):
        return tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True, tokenize=False)
    return str(prompt_obj)


def _read_input_questions(args: argparse.Namespace, reward_module: Any) -> list[QuestionRecord]:
    table = pq.read_table(args.input_parquet, columns=["prompt", "reward_model", "extra_info"])
    rows = table.to_pylist()
    if args.question_limit > 0:
        rows = rows[: args.question_limit]

    rank_start = max(int(args.question_rank_start), 0)
    rank_end = int(args.question_rank_end)
    indexed_rows = [
        (idx, row)
        for idx, row in enumerate(rows)
        if idx >= rank_start and (rank_end < 0 or idx < rank_end)
    ]

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)

    questions: list[QuestionRecord] = []
    seen_qid: set[str] = set()
    for idx, row in indexed_rows:
        extra_info = row.get("extra_info") or {}
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") or {}
        qid = str(extra_info.get("question_id", f"row_{idx}"))
        if qid in seen_qid:
            raise RuntimeError(f"duplicate question_id in input: {qid}")
        seen_qid.add(qid)

        prompt_obj = row.get("prompt")
        prompt_text = _format_prompt(tokenizer, prompt_obj)
        target_set = _extract_target_set(reward_module, ground_truth)
        choice_set = _extract_choice_set(reward_module, ground_truth)
        questions.append(
            QuestionRecord(
                question_id=qid,
                question_rank=idx,
                split=args.split,
                prompt_id=f"{args.split}:{qid}",
                prompt_obj=prompt_obj,
                prompt_text=prompt_text,
                target_set=target_set,
                choice_set=choice_set,
            )
        )
    return questions


def _read_existing_probe_map(parsed_dir: Path) -> dict[str, set[int]]:
    probe_map: dict[str, set[int]] = defaultdict(set)
    if not parsed_dir.exists():
        return probe_map
    for path in sorted(parsed_dir.glob("part-*.parquet")):
        t = pq.ParquetFile(path).read(columns=["question_id", "probe_index"])
        qids = t.column("question_id").to_pylist()
        probes = t.column("probe_index").to_pylist()
        for qid, probe in zip(qids, probes):
            if qid is None or probe is None:
                continue
            probe_map[str(qid)].add(int(probe))
    return probe_map


def _chunked(values: list[int], n: int) -> list[list[int]]:
    return [values[i : i + n] for i in range(0, len(values), n)]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _write_worker_progress(
    progress_path: Path,
    worker_id: int,
    questions_assigned: int,
    questions_completed: int,
    generated_rows: int,
) -> None:
    _write_json_atomic(
        progress_path,
        {
            "worker_id": int(worker_id),
            "questions_assigned": int(questions_assigned),
            "questions_completed": int(questions_completed),
            "questions_remaining": int(max(questions_assigned - questions_completed, 0)),
            "generated_rows": int(generated_rows),
            "updated_at_utc": _utc_now(),
        },
    )


def _read_progress_totals(tmp_root: Path, num_workers: int) -> dict[str, int]:
    questions_assigned = 0
    questions_completed = 0
    generated_rows = 0
    for worker_id in range(num_workers):
        path = tmp_root / f"worker_{worker_id:02d}" / "progress.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        questions_assigned += int(payload.get("questions_assigned", 0) or 0)
        questions_completed += int(payload.get("questions_completed", 0) or 0)
        generated_rows += int(payload.get("generated_rows", 0) or 0)
    return {
        "questions_assigned": int(questions_assigned),
        "questions_completed": int(questions_completed),
        "questions_remaining": int(max(questions_assigned - questions_completed, 0)),
        "generated_rows": int(generated_rows),
    }


def _sample_gpu_utilization(gpu_ids: list[int]) -> dict[int, float]:
    if not gpu_ids:
        return {}
    cmd = [
        "nvidia-smi",
        f"--id={','.join(str(x) for x in gpu_ids)}",
        "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return {}
    out: dict[int, float] = {}
    for line in proc.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [x.strip() for x in s.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpu_idx = int(parts[0])
            util = float(parts[1])
        except Exception:
            continue
        out[gpu_idx] = util
    return out


def _print_monitor_snapshot(
    tmp_root: Path,
    num_workers: int,
    total_questions: int,
    gpu_window_sum: dict[int, float],
    gpu_window_count: dict[int, int],
    window_start_ts: float,
    tag: str,
) -> None:
    totals = _read_progress_totals(tmp_root=tmp_root, num_workers=num_workers)
    assigned = int(totals["questions_assigned"] or total_questions)
    done = int(totals["questions_completed"])
    remaining = int(max(assigned - done, 0))
    generated_rows = int(totals["generated_rows"])

    total_gpu_samples = int(sum(gpu_window_count.values()))
    if total_gpu_samples > 0:
        total_gpu_sum = float(sum(gpu_window_sum.values()))
        gpu_avg_all = total_gpu_sum / total_gpu_samples
        per_gpu_avg = ", ".join(
            f"{gid}:{(gpu_window_sum[gid] / gpu_window_count[gid]):.2f}%"
            for gid in sorted(gpu_window_count.keys())
            if gpu_window_count[gid] > 0
        )
    else:
        gpu_avg_all = None
        per_gpu_avg = "NA"

    window_seconds = int(max(time.time() - window_start_ts, 0.0))
    gpu_avg_str = f"{gpu_avg_all:.2f}%" if gpu_avg_all is not None else "NA"
    print(
        f"[monitor:{tag}] window_s={window_seconds} gpu_util_avg_all={gpu_avg_str} "
        f"gpu_util_avg_by_id={per_gpu_avg} questions_done={done}/{assigned} "
        f"questions_remaining={remaining} generated_rows={generated_rows}",
        flush=True,
    )


class ParquetPartWriter:
    def __init__(self, out_dir: Path, prefix: str, rows_per_part: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.rows_per_part = rows_per_part
        self.rows: list[dict[str, Any]] = []
        self.part_idx = 0
        self.total_rows = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_part:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        out_path = self.out_dir / f"{self.prefix}{self.part_idx:05d}.parquet"
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, out_path, compression="zstd")
        self.total_rows += len(self.rows)
        self.rows = []
        self.part_idx += 1

    def close(self) -> None:
        self.flush()


def _extract_final_boxed(reward_module: Any, response_text: str) -> tuple[bool, str]:
    parsed = reward_module.parse_medical_answer(
        response_text,
        policy=reward_module.POLICY_FINAL_ANSWER_LAST_BOXED,
    )
    if parsed.parse_status == "no_boxed":
        return False, ""
    return True, parsed.boxed_raw


def _parse_response(
    reward_module: Any,
    response_text: str,
    target_set: set[str],
    choice_set: set[str],
) -> dict[str, Any]:
    parsed = reward_module.parse_medical_answer(
        response_text,
        policy=reward_module.POLICY_FINAL_ANSWER_LAST_BOXED,
        choice_set=choice_set,
    )
    base = {
        "boxed_raw": parsed.boxed_raw,
        "normalized_choice": parsed.normalized_choice,
        "is_abstain": int(parsed.is_abstain),
        "is_correct": 0,
        "parse_status": parsed.parse_status,
        "parser_policy": parsed.policy,
        "boxed_count": int(parsed.boxed_count),
        "used_answer_block": int(parsed.used_answer_block),
    }

    if parsed.parse_status == "ok_choice":
        base["is_correct"] = int(parsed.normalized_choice in target_set)
    return base


def _worker_run(
    worker_id: int,
    gpu_id: int,
    worker_questions: list[dict[str, Any]],
    existing_probe_map: dict[str, list[int]],
    tmp_root: Path,
    config: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams

    reward_module = _load_reward_module(Path(config["repo_root"]))

    worker_dir = tmp_root / f"worker_{worker_id:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    progress_path = worker_dir / "progress.json"
    responses_writer = ParquetPartWriter(worker_dir / "responses", f"w{worker_id:02d}_part-", int(config["part_rows"]))
    parsed_writer = ParquetPartWriter(worker_dir / "parsed", f"w{worker_id:02d}_part-", int(config["part_rows"]))

    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        max_model_len=int(config["max_model_len"]),
        max_num_seqs=int(config["max_num_seqs"]),
        max_num_batched_tokens=int(config["max_num_batched_tokens"]),
    )

    total_generated = 0
    total_questions = len(worker_questions)
    t0 = time.time()
    _write_worker_progress(
        progress_path=progress_path,
        worker_id=worker_id,
        questions_assigned=total_questions,
        questions_completed=0,
        generated_rows=0,
    )

    for q_idx, q in enumerate(worker_questions):
        qid = q["question_id"]
        target_set = set(q["target_set"])
        choice_set = set(q["choice_set"])
        existing = set(int(x) for x in existing_probe_map.get(qid, []))
        missing = [i for i in range(int(config["probes_per_question"])) if i not in existing]

        if not missing:
            _write_worker_progress(
                progress_path=progress_path,
                worker_id=worker_id,
                questions_assigned=total_questions,
                questions_completed=q_idx + 1,
                generated_rows=total_generated,
            )
            continue

        for probe_indices in _chunked(missing, int(config["n_chunk"])):
            stop_list = [config["stop_str"]] if config["stop_str"] else []
            base_sampling_kwargs = dict(
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                top_k=int(config["top_k"]),
                min_p=float(config["min_p"]),
                max_tokens=int(config["max_new_tokens"]),
                stop=stop_list,
                include_stop_str_in_output=bool(config["include_stop_str_in_output"]),
                skip_special_tokens=True,
            )

            # Adaptive fallback for potential OOM/transient generation errors.
            stack = [probe_indices]
            while stack:
                current = stack.pop(0)
                call_seed = int(config["seed_base"]) + int(q["question_rank"]) * 100000 + int(current[0])
                params = SamplingParams(
                    n=len(current),
                    seed=call_seed,
                    **base_sampling_kwargs,
                )
                try:
                    outputs = llm.generate([q["prompt_text"]], params, use_tqdm=False)
                except Exception:
                    if len(current) == 1:
                        raise
                    mid = len(current) // 2
                    stack.insert(0, current[mid:])
                    stack.insert(0, current[:mid])
                    continue

                first = outputs[0]
                if len(first.outputs) != len(current):
                    raise RuntimeError(
                        f"worker={worker_id} question_id={qid} expected {len(current)} outputs, got {len(first.outputs)}"
                    )

                for offset, out in enumerate(first.outputs):
                    probe_index = int(current[offset])
                    response_text = out.text if isinstance(out.text, str) else str(out.text)
                    if response_text == "":
                        raise RuntimeError(
                            f"worker={worker_id} question_id={qid} probe_index={probe_index} produced empty response"
                        )

                    created_at_utc = _utc_now()
                    finish_reason = str(getattr(out, "finish_reason", "unknown"))
                    token_ids = getattr(out, "token_ids", None)
                    generated_tokens = int(len(token_ids)) if token_ids is not None else 0
                    response_sha256 = hashlib.sha256(response_text.encode("utf-8")).hexdigest()

                    parsed = _parse_response(
                        reward_module=reward_module,
                        response_text=response_text,
                        target_set=target_set,
                        choice_set=choice_set,
                    )

                    responses_writer.add(
                        {
                            "run_id": config["run_id"],
                            "split": config["split"],
                            "question_id": qid,
                            "prompt_id": q["prompt_id"],
                            "probe_index": probe_index,
                            "response_text": response_text,
                            "response_sha256": response_sha256,
                            "model_id": config["model_id"],
                            "temperature": float(config["temperature"]),
                            "top_p": float(config["top_p"]),
                            "seed": int(params.seed),
                            "created_at_utc": created_at_utc,
                            "finish_reason": finish_reason,
                            "generated_tokens": generated_tokens,
                        }
                    )
                    parsed_writer.add(
                        {
                            "question_id": qid,
                            "probe_index": probe_index,
                            "boxed_raw": parsed["boxed_raw"],
                            "normalized_choice": parsed["normalized_choice"],
                            "is_abstain": int(parsed["is_abstain"]),
                            "is_correct": int(parsed["is_correct"]),
                            "parse_status": parsed["parse_status"],
                            "parser_policy": parsed["parser_policy"],
                            "boxed_count": int(parsed["boxed_count"]),
                            "used_answer_block": int(parsed["used_answer_block"]),
                        }
                    )
                    total_generated += 1

        _write_worker_progress(
            progress_path=progress_path,
            worker_id=worker_id,
            questions_assigned=total_questions,
            questions_completed=q_idx + 1,
            generated_rows=total_generated,
        )

        if (q_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"[worker {worker_id}] progress={q_idx + 1}/{total_questions} generated={total_generated} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    responses_writer.close()
    parsed_writer.close()
    _write_worker_progress(
        progress_path=progress_path,
        worker_id=worker_id,
        questions_assigned=total_questions,
        questions_completed=total_questions,
        generated_rows=total_generated,
    )

    summary = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "questions_assigned": total_questions,
        "generated_rows": total_generated,
        "responses_rows_written": responses_writer.total_rows,
        "parsed_rows_written": parsed_writer.total_rows,
        "finished_at_utc": _utc_now(),
    }
    (worker_dir / "worker_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")



def _prepare_dirs(run_dir: Path, resume: bool) -> dict[str, Path]:
    responses_dir = run_dir / "responses"
    parsed_dir = run_dir / "parsed"
    labels_dir = run_dir / "labels"
    reports_dir = run_dir / "reports"

    if run_dir.exists() and not resume:
        raise FileExistsError(f"run dir already exists: {run_dir}; pass --resume to continue")

    run_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "responses_split_dir": responses_dir,
        "parsed_split_dir": parsed_dir,
        "labels_dir": labels_dir,
        "reports_dir": reports_dir,
    }


def _existing_part_index_max(split_dir: Path) -> int:
    max_idx = -1
    for p in split_dir.glob("part-*.parquet"):
        stem = p.stem
        try:
            idx = int(stem.split("-")[-1])
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    return max_idx


def _consolidate_worker_parts(tmp_root: Path, final_split_dir: Path, subdir: str) -> int:
    final_split_dir.mkdir(parents=True, exist_ok=True)
    tmp_files: list[Path] = []
    for worker_dir in sorted(tmp_root.glob("worker_*")):
        tmp_files.extend(sorted((worker_dir / subdir).glob("w*_part-*.parquet")))

    next_idx = _existing_part_index_max(final_split_dir) + 1
    moved = 0
    for src in tmp_files:
        dst = final_split_dir / f"part-{next_idx:05d}.parquet"
        src.rename(dst)
        next_idx += 1
        moved += 1
    return moved


def _load_part_files(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("part-*.parquet"))


def _count_rows(part_files: list[Path]) -> int:
    total = 0
    for p in part_files:
        meta = pq.read_metadata(p)
        total += int(meta.num_rows)
    return total


def _analyze_responses(responses_parts: list[Path]) -> tuple[int, int]:
    row_count = 0
    empty_count = 0
    for p in responses_parts:
        t = pq.ParquetFile(p).read(columns=["response_text"])
        n = int(t.num_rows)
        row_count += n

        col = t.column("response_text")
        null_mask = pc.is_null(col)
        str_len = pc.utf8_length(col)
        zero_mask = pc.equal(str_len, 0)
        bad_mask = pc.or_(null_mask, zero_mask)
        bad = pc.sum(pc.cast(bad_mask, pa.int64())).as_py()
        empty_count += int(bad or 0)
    return row_count, empty_count


def _analyze_parsed(
    parsed_parts: list[Path],
) -> tuple[int, dict[str, set[int]], dict[str, int], dict[str, int], Counter[str]]:
    row_count = 0
    probe_map: dict[str, set[int]] = defaultdict(set)
    correct_map: dict[str, int] = defaultdict(int)
    abstain_map: dict[str, int] = defaultdict(int)
    parse_status = Counter()

    for p in parsed_parts:
        t = pq.ParquetFile(p).read(columns=["question_id", "probe_index", "is_correct", "is_abstain", "parse_status"])
        row_count += int(t.num_rows)
        qids = t.column("question_id").to_pylist()
        probes = t.column("probe_index").to_pylist()
        corrects = t.column("is_correct").to_pylist()
        abstains = t.column("is_abstain").to_pylist()
        statuses = t.column("parse_status").to_pylist()

        for qid, probe, corr, abst, status in zip(qids, probes, corrects, abstains, statuses):
            q = str(qid)
            probe_map[q].add(int(probe))
            correct_map[q] += int(corr or 0)
            abstain_map[q] += int(abst or 0)
            parse_status[str(status)] += 1

    return row_count, probe_map, correct_map, abstain_map, parse_status


def _build_labels(
    questions: list[QuestionRecord],
    probe_map: dict[str, set[int]],
    correct_map: dict[str, int],
    abstain_map: dict[str, int],
    probes_per_question: int,
    ook_rule_version: str,
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for q in questions:
        probes = probe_map.get(q.question_id, set())
        correct_count = int(correct_map.get(q.question_id, 0))
        abstain_count = int(abstain_map.get(q.question_id, 0))
        labels.append(
            {
                "question_id": q.question_id,
                "probe_count": int(len(probes)),
                "correct_count": correct_count,
                "abstain_count": abstain_count,
                "ook": bool(correct_count == 0),
                "ook_rule_version": ook_rule_version,
            }
        )
    return labels


def _validate(
    questions: list[QuestionRecord],
    probe_map: dict[str, set[int]],
    responses_row_count: int,
    parsed_row_count: int,
    empty_response_count: int,
    probes_per_question: int,
) -> dict[str, Any]:
    expected = len(questions) * probes_per_question
    expected_probe_set = set(range(probes_per_question))

    probe_count_ok = True
    coverage_ok = True
    min_probe = probes_per_question
    max_probe = 0
    missing_examples: list[dict[str, Any]] = []

    for q in questions:
        probes = probe_map.get(q.question_id, set())
        c = len(probes)
        min_probe = min(min_probe, c)
        max_probe = max(max_probe, c)
        if c != probes_per_question:
            probe_count_ok = False
        if probes != expected_probe_set:
            coverage_ok = False
            if len(missing_examples) < 5:
                missing = sorted(expected_probe_set - probes)
                missing_examples.append({"question_id": q.question_id, "missing_probe_index_head": missing[:10]})

    empty_ratio = (empty_response_count / responses_row_count) if responses_row_count > 0 else 1.0

    checks = {
        "row_count_ok": responses_row_count == expected and parsed_row_count == expected,
        "probe_count_per_question_ok": probe_count_ok,
        "probe_index_coverage_ok": coverage_ok,
        "empty_response_ok": empty_response_count == 0,
        "parsed_matches_responses": parsed_row_count == responses_row_count,
    }
    overall_pass = all(checks.values())

    return {
        "expected_row_count": expected,
        "responses_row_count": responses_row_count,
        "parsed_row_count": parsed_row_count,
        "empty_response_count": empty_response_count,
        "empty_response_ratio": empty_ratio,
        "probe_count_min": min_probe,
        "probe_count_max": max_probe,
        "missing_probe_examples": missing_examples,
        "checks": checks,
        "overall_pass": overall_pass,
    }


def _write_quality_report_md(
    out_path: Path,
    run_id: str,
    split: str,
    validation: dict[str, Any],
    parse_status: Counter[str],
    labels: list[dict[str, Any]],
    started_at: str,
    ended_at: str,
) -> None:
    ook_count = sum(1 for x in labels if x["ook"])
    abstain_total = sum(int(x["abstain_count"]) for x in labels)
    correct_total = sum(int(x["correct_count"]) for x in labels)
    total_rows = int(validation["responses_row_count"])

    lines = [
        "# KBP Quality Report",
        "",
        f"- run_id: `{run_id}`",
        f"- split: `{split}`",
        f"- started_at_utc: `{started_at}`",
        f"- ended_at_utc: `{ended_at}`",
        "",
        "## Validation",
        "",
        f"- overall_pass: `{validation['overall_pass']}`",
        f"- expected_row_count: `{validation['expected_row_count']}`",
        f"- responses_row_count: `{validation['responses_row_count']}`",
        f"- parsed_row_count: `{validation['parsed_row_count']}`",
        f"- empty_response_count: `{validation['empty_response_count']}`",
        f"- empty_response_ratio: `{validation['empty_response_ratio']:.8f}`",
        f"- probe_count_min: `{validation['probe_count_min']}`",
        f"- probe_count_max: `{validation['probe_count_max']}`",
        "",
        "## Parse Status",
        "",
    ]

    for k in sorted(parse_status.keys()):
        lines.append(f"- `{k}`: `{parse_status[k]}`")

    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            f"- ook_question_count: `{ook_count}`",
            f"- abstain_total: `{abstain_total}`",
            f"- correct_total: `{correct_total}`",
            f"- abstain_rate: `{(abstain_total / total_rows) if total_rows > 0 else 0.0:.8f}`",
            f"- correct_rate: `{(correct_total / total_rows) if total_rows > 0 else 0.0:.8f}`",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_utc = _utc_now()
    repo_root = Path(__file__).resolve().parents[2]

    if args.probes_per_question <= 0:
        raise ValueError("--probes_per_question must be > 0")
    if args.n_chunk <= 0:
        raise ValueError("--n_chunk must be > 0")
    if args.probes_per_question % args.n_chunk != 0 and not args.resume:
        raise ValueError("--probes_per_question should be divisible by --n_chunk for fresh runs")
    if args.question_rank_start < 0:
        raise ValueError("--question_rank_start must be >= 0")
    if args.question_rank_end != -1 and args.question_rank_end <= args.question_rank_start:
        raise ValueError("--question_rank_end must be -1 or > --question_rank_start")

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpu_ids parsed empty")
    if args.num_workers != len(gpu_ids):
        raise ValueError("--num_workers must equal number of --gpu_ids entries")

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"input parquet not found: {args.input_parquet}")
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"model path invalid: {args.model_path}")

    reward_module = _load_reward_module(repo_root)
    questions = _read_input_questions(args, reward_module)
    if not questions:
        raise RuntimeError("no questions selected")

    run_dir = args.output_root / "runs" / args.run_id
    dirs = _prepare_dirs(run_dir, args.resume)

    responses_split_dir = dirs["responses_split_dir"] / f"split={args.split}"
    parsed_split_dir = dirs["parsed_split_dir"] / f"split={args.split}"
    responses_split_dir.mkdir(parents=True, exist_ok=True)
    parsed_split_dir.mkdir(parents=True, exist_ok=True)

    existing_probe_map = _read_existing_probe_map(parsed_split_dir) if args.resume else {}

    prompts_path = run_dir / "prompts.parquet"
    if not prompts_path.exists():
        prompt_rows = [
            {
                "run_id": args.run_id,
                "split": args.split,
                "question_id": q.question_id,
                "prompt_id": q.prompt_id,
                "question_rank": q.question_rank,
                "prompt": q.prompt_obj,
            }
            for q in questions
        ]
        pq.write_table(pa.Table.from_pylist(prompt_rows), prompts_path, compression="zstd")

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": args.run_id,
        "stage": "D2.5",
        "status": "running",
        "created_at_utc": start_utc,
        "input_parquet": str(args.input_parquet),
        "output_run_dir": str(run_dir),
        "split": args.split,
        "parser": {
            "policy": reward_module.POLICY_FINAL_ANSWER_LAST_BOXED,
            "module": "verl.utils.reward_score.medical_answer_parser",
        },
        "question_count": len(questions),
        "expected_row_count": len(questions) * args.probes_per_question,
        "model_path": str(args.model_path),
        "model_id": args.model_id,
        "sampling": {
            "probes_per_question": args.probes_per_question,
            "n_chunk": args.n_chunk,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
            "stop_str": args.stop_str,
            "include_stop_str_in_output": args.include_stop_str_in_output,
            "seed_base": args.seed_base,
        },
        "runtime": {
            "num_workers": args.num_workers,
            "gpu_ids": gpu_ids,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "question_rank_start": args.question_rank_start,
            "question_rank_end": args.question_rank_end,
            "resume": args.resume,
            "postprocess_only": args.postprocess_only,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    tmp_root: Path | None = None
    moved_resp = 0
    moved_parsed = 0

    if not args.postprocess_only:
        tmp_root = run_dir / "_tmp" / f"session_{int(time.time())}"
        tmp_root.mkdir(parents=True, exist_ok=True)

        worker_questions: list[list[dict[str, Any]]] = [[] for _ in range(args.num_workers)]
        for q in questions:
            w = q.question_rank % args.num_workers
            worker_questions[w].append(
                {
                    "question_id": q.question_id,
                    "question_rank": q.question_rank,
                    "prompt_id": q.prompt_id,
                    "prompt_text": q.prompt_text,
                    "target_set": sorted(q.target_set),
                    "choice_set": sorted(q.choice_set),
                }
            )

        existing_probe_map_lists = {k: sorted(v) for k, v in existing_probe_map.items()}

        config = {
            "repo_root": str(repo_root),
            "run_id": args.run_id,
            "split": args.split,
            "model_path": str(args.model_path),
            "model_id": args.model_id,
            "probes_per_question": args.probes_per_question,
            "n_chunk": args.n_chunk,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
            "stop_str": args.stop_str,
            "include_stop_str_in_output": args.include_stop_str_in_output,
            "seed_base": args.seed_base,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "part_rows": args.part_rows,
        }

        mp.set_start_method("spawn", force=True)
        procs: list[mp.Process] = []
        for worker_id in range(args.num_workers):
            p = mp.Process(
                target=_worker_run,
                args=(
                    worker_id,
                    gpu_ids[worker_id],
                    worker_questions[worker_id],
                    existing_probe_map_lists,
                    tmp_root,
                    config,
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)

        monitor_window_start = time.time()
        next_sample_at = monitor_window_start
        next_report_at = monitor_window_start + MONITOR_REPORT_INTERVAL_S
        gpu_window_sum: dict[int, float] = defaultdict(float)
        gpu_window_count: dict[int, int] = defaultdict(int)

        while True:
            now_ts = time.time()
            if now_ts >= next_sample_at:
                samples = _sample_gpu_utilization(gpu_ids)
                for gpu_idx, util in samples.items():
                    gpu_window_sum[int(gpu_idx)] += float(util)
                    gpu_window_count[int(gpu_idx)] += 1
                while next_sample_at <= now_ts:
                    next_sample_at += MONITOR_SAMPLE_INTERVAL_S

            if now_ts >= next_report_at:
                _print_monitor_snapshot(
                    tmp_root=tmp_root,
                    num_workers=args.num_workers,
                    total_questions=len(questions),
                    gpu_window_sum=gpu_window_sum,
                    gpu_window_count=gpu_window_count,
                    window_start_ts=monitor_window_start,
                    tag="10m",
                )
                gpu_window_sum.clear()
                gpu_window_count.clear()
                monitor_window_start = now_ts
                while next_report_at <= now_ts:
                    next_report_at += MONITOR_REPORT_INTERVAL_S

            if not any(p.is_alive() for p in procs):
                break
            time.sleep(1.0)

        for p in procs:
            p.join()

        _print_monitor_snapshot(
            tmp_root=tmp_root,
            num_workers=args.num_workers,
            total_questions=len(questions),
            gpu_window_sum=gpu_window_sum,
            gpu_window_count=gpu_window_count,
            window_start_ts=monitor_window_start,
            tag="final",
        )

        failed_workers = [p.pid for p in procs if p.exitcode != 0]
        if failed_workers:
            raise RuntimeError(f"worker failed, pids={failed_workers}")

        moved_resp = _consolidate_worker_parts(tmp_root, responses_split_dir, "responses")
        moved_parsed = _consolidate_worker_parts(tmp_root, parsed_split_dir, "parsed")

    responses_parts = _load_part_files(responses_split_dir)
    parsed_parts = _load_part_files(parsed_split_dir)
    if not responses_parts or not parsed_parts:
        raise RuntimeError("missing responses/parsed parts; cannot continue postprocess")

    responses_row_count, empty_response_count = _analyze_responses(responses_parts)
    parsed_row_count, probe_map, correct_map, abstain_map, parse_status_counter = _analyze_parsed(parsed_parts)

    labels = _build_labels(
        questions=questions,
        probe_map=probe_map,
        correct_map=correct_map,
        abstain_map=abstain_map,
        probes_per_question=args.probes_per_question,
        ook_rule_version=args.ook_rule_version,
    )

    labels_path = dirs["labels_dir"] / "kbp_labels_by_question.parquet"
    pq.write_table(pa.Table.from_pylist(labels), labels_path, compression="zstd")

    labels_check = pq.ParquetFile(labels_path).read().to_pylist()
    label_by_q = {str(x["question_id"]): x for x in labels}
    labels_match = True
    for row in labels_check:
        qid = str(row["question_id"])
        src = label_by_q.get(qid)
        if src is None:
            labels_match = False
            break
        for key in ["probe_count", "correct_count", "abstain_count", "ook", "ook_rule_version"]:
            if row[key] != src[key]:
                labels_match = False
                break

    validation = _validate(
        questions=questions,
        probe_map=probe_map,
        responses_row_count=responses_row_count,
        parsed_row_count=parsed_row_count,
        empty_response_count=empty_response_count,
        probes_per_question=args.probes_per_question,
    )
    validation["checks"]["summary_recomputable_ok"] = labels_match
    validation["overall_pass"] = validation["overall_pass"] and bool(labels_match)

    ended_at = _utc_now()
    duration_s = (
        datetime.fromisoformat(ended_at).timestamp() - datetime.fromisoformat(start_utc).timestamp()
    )

    summary = {
        "run_id": args.run_id,
        "stage": "D2.5",
        "status": "pass" if validation["overall_pass"] else "fail",
        "started_at_utc": start_utc,
        "ended_at_utc": ended_at,
        "duration_seconds": duration_s,
        "input_parquet": str(args.input_parquet),
        "split": args.split,
        "question_count": len(questions),
        "sampling": manifest["sampling"],
        "runtime": manifest["runtime"],
        "files": {
            "manifest": str(manifest_path),
            "prompts": str(prompts_path),
            "responses_dir": str(responses_split_dir),
            "parsed_dir": str(parsed_split_dir),
            "labels": str(labels_path),
            "reports_dir": str(dirs["reports_dir"]),
            "responses_part_count": len(responses_parts),
            "parsed_part_count": len(parsed_parts),
            "moved_response_parts": moved_resp,
            "moved_parsed_parts": moved_parsed,
        },
        "validation": validation,
        "parser": {
            "policy": reward_module.POLICY_FINAL_ANSWER_LAST_BOXED,
            "module": "verl.utils.reward_score.medical_answer_parser",
        },
        "parse_status_counts": dict(parse_status_counter),
        "label_stats": {
            "ook_question_count": int(sum(1 for x in labels if x["ook"])),
            "abstain_total": int(sum(int(x["abstain_count"]) for x in labels)),
            "correct_total": int(sum(int(x["correct_count"]) for x in labels)),
        },
    }

    summary_path = dirs["reports_dir"] / "kbp_summary.json"
    quality_path = dirs["reports_dir"] / "kbp_quality_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_quality_report_md(
        out_path=quality_path,
        run_id=args.run_id,
        split=args.split,
        validation=validation,
        parse_status=parse_status_counter,
        labels=labels,
        started_at=start_utc,
        ended_at=ended_at,
    )

    manifest["status"] = summary["status"]
    manifest["ended_at_utc"] = ended_at
    manifest["summary_path"] = str(summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if validation["overall_pass"] and tmp_root is not None:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[INFO] run_id={args.run_id}")
    print(f"[INFO] responses_row_count={responses_row_count}")
    print(f"[INFO] parsed_row_count={parsed_row_count}")
    print(f"[INFO] empty_response_count={empty_response_count}")
    print(f"[INFO] summary={summary_path}")
    print(f"[INFO] quality={quality_path}")

    if not validation["overall_pass"]:
        raise SystemExit("KBP validation failed. See summary report for details.")


if __name__ == "__main__":
    main()
