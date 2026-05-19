#!/usr/bin/env python3
"""Host-side preflight checks for EMBC medical TruthRL runs.

This script intentionally avoids importing trainer internals or touching model
weights. It validates paths, import resolution, dataset schemas when requested,
and Slurm/script launch surfaces before a run is submitted.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_EXPECTED_ACCOUNT = "def-zshakeri"

DEFAULT_RL_FILES = [
    "data/medical/verl/medqa_grpo_train.parquet",
    "data/medical/verl/medqa_grpo_test.parquet",
    "data/medical/verl/medmcqa_eval.parquet",
]
DEFAULT_SFT_FILES = [
    "data/medical/verl/medqa_sft_train.parquet",
    "data/medical/verl/medqa_sft_val.parquet",
]
DEFAULT_SHELL_GLOBS = [
    "scripts/medical/run_staged_d*.sh",
    "scripts/medical/run_kbp_medqa.sh",
    "scripts/medical/run_kbp_merge_and_postprocess.sh",
    "scripts/medical/submit_kbp_medqa_parallel.sh",
    "scripts/medical/validate_medical_data.sh",
]
DEFAULT_DRY_RUN_GLOBS = [
    "scripts/medical/run_staged_d1_grpo_truthrl.sh",
    "scripts/medical/run_staged_d2_grpo_truthrl_cpro.sh",
    "scripts/medical/run_staged_d3_hard_ook_truthrl.sh",
    "scripts/medical/run_kbp_medqa.sh",
]
DEFAULT_SLURM_GLOBS = [
    "scripts/medical/submit_staged_d*.slurm",
    "scripts/medical/submit_stageD*.slurm",
    "scripts/medical/submit_kbp_medqa_full.slurm",
    "scripts/medical/submit_kbp_medqa_merge.slurm",
]
DRIFT_PATTERNS = {
    "old_home_project_path": "/home/erichyu/projects/def-zshakeri/erichyu/embc",
    "old_home_scratch_embc": "/home/erichyu/scratch/embc",
    "old_gpu_account": "def-zshakeri_gpu",
}


@dataclass
class CheckResult:
    name: str
    status: str = "pass"
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == "pass":
            self.status = "warn"

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.status = "fail"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        _resolve(path).relative_to(_resolve(parent))
        return True
    except ValueError:
        return False


def _as_abs(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root / p


def _git_output(repo_root: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"git unavailable: {exc}"
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip()
    return proc.stdout.strip()


def discover_files(repo_root: Path, globs: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(repo_root.glob(pattern)):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    return files


def check_paths(
    repo_root: Path,
    project_root: Path,
    cpro_root: Path,
    model_path: Path,
    rl_files: list[Path],
    sft_files: list[Path],
    strict_files: bool,
) -> CheckResult:
    result = CheckResult("paths")
    expected_dirs = [
        repo_root / "scripts" / "medical",
        repo_root / "training" / "verl" / "verl",
        repo_root / "data_utils" / "medical",
    ]
    for path in expected_dirs:
        if not path.is_dir():
            result.fail(f"Missing expected directory: {path}")

    if not (repo_root / ".git").exists():
        result.fail(f"TruthRL repo root does not contain .git: {repo_root}")
    if not cpro_root.is_dir():
        result.warn(f"CPRO root not found: {cpro_root}")
    elif not (cpro_root / "verl").is_dir():
        result.warn(f"CPRO root exists but does not contain CPRO/verl: {cpro_root}")

    model_config = model_path / "config.json"
    if not model_config.is_file():
        message = f"MODEL_PATH missing config.json: {model_path}"
        if strict_files:
            result.fail(message)
        else:
            result.warn(message)

    missing_rl = [str(p) for p in rl_files if not p.is_file()]
    missing_sft = [str(p) for p in sft_files if not p.is_file()]
    if missing_rl or missing_sft:
        message = {"missing_rl_files": missing_rl, "missing_sft_files": missing_sft}
        if strict_files:
            result.fail(f"Missing parquet files: {message}")
        else:
            result.warn(f"Some parquet files are missing: {message}")

    image_path = project_root / "envs" / "truthrl_apptainer" / "truthrl.sif"
    if not image_path.is_file():
        result.warn(f"Default Apptainer image not found: {image_path}")

    result.details.update(
        {
            "repo_root": str(repo_root),
            "project_root": str(project_root),
            "cpro_root": str(cpro_root),
            "model_path": str(model_path),
            "default_apptainer_image": str(image_path),
            "rl_files": [str(p) for p in rl_files],
            "sft_files": [str(p) for p in sft_files],
        }
    )
    return result


def _find_verl_spec(search_path: list[str] | None = None) -> str | None:
    spec = importlib.machinery.PathFinder.find_spec("verl", search_path)
    if spec is None:
        return None
    return spec.origin or "<namespace>"


def check_verl_import(repo_root: Path, cpro_root: Path, strict_ambient: bool) -> CheckResult:
    result = CheckResult("verl_import")
    expected_parent = repo_root / "training" / "verl"
    expected_pkg = expected_parent / "verl"
    origin = _find_verl_spec([str(expected_parent)])
    result.details["expected_pythonpath"] = str(expected_parent)
    result.details["expected_spec_origin"] = origin

    if origin is None:
        result.fail(f"Cannot resolve verl from expected path: {expected_parent}")
    elif not _is_relative_to(Path(origin), expected_pkg):
        result.fail(f"Expected verl resolves outside TruthRL training/verl: {origin}")

    ambient_origin = _find_verl_spec()
    result.details["ambient_spec_origin"] = ambient_origin
    if ambient_origin and ambient_origin != "<namespace>" and not _is_relative_to(Path(ambient_origin), expected_pkg):
        message = f"Ambient python can resolve a non-TruthRL verl package: {ambient_origin}"
        if strict_ambient:
            result.fail(message)
        else:
            result.warn(message)

    pythonpath_entries = [Path(x) for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]
    cpro_verl = cpro_root / "verl"
    truthrl_index = None
    cpro_index = None
    for idx, entry in enumerate(pythonpath_entries):
        resolved = _resolve(entry)
        if resolved == _resolve(expected_parent) and truthrl_index is None:
            truthrl_index = idx
        if resolved == _resolve(cpro_verl) and cpro_index is None:
            cpro_index = idx
    result.details["pythonpath"] = [str(p) for p in pythonpath_entries]
    if cpro_index is not None and (truthrl_index is None or cpro_index < truthrl_index):
        result.fail(
            "PYTHONPATH would prefer CPRO/verl over TruthRL/training/verl; "
            f"CPRO index={cpro_index}, TruthRL index={truthrl_index}"
        )

    return result


def check_schema(repo_root: Path, rl_files: list[Path], sft_files: list[Path], mode: str) -> CheckResult:
    result = CheckResult("schema")
    result.details["mode"] = mode
    if mode == "skip":
        result.status = "skip"
        return result

    existing_rl = [p for p in rl_files if p.is_file()]
    existing_sft = [p for p in sft_files if p.is_file()]
    if mode == "strict":
        for path in rl_files + sft_files:
            if not path.is_file():
                result.fail(f"Missing schema input file: {path}")
    elif not existing_rl and not existing_sft:
        result.warn("No existing parquet files found for schema validation.")
        return result

    try:
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from data_utils.medical import validate_medical_data
    except BaseException as exc:
        message = f"Cannot import data schema validator, likely missing pyarrow: {exc}"
        if mode == "strict":
            result.fail(message)
        else:
            result.warn(message)
        return result

    rl_summaries = []
    sft_summaries = []
    for path in existing_rl:
        try:
            rl_summaries.append(asdict(validate_medical_data.validate_rl_schema(path)))
        except Exception as exc:
            result.fail(f"RL schema validation failed for {path}: {exc}")
    for path in existing_sft:
        try:
            sft_summaries.append(asdict(validate_medical_data.validate_sft_schema(path)))
        except Exception as exc:
            result.fail(f"SFT schema validation failed for {path}: {exc}")

    result.details["rl_summaries"] = rl_summaries
    result.details["sft_summaries"] = sft_summaries
    return result


def check_bash_syntax(paths: list[Path]) -> CheckResult:
    result = CheckResult("script_syntax")
    checked = []
    for path in paths:
        proc = subprocess.run(["bash", "-n", str(path)], check=False, capture_output=True, text=True)
        checked.append(str(path))
        if proc.returncode != 0:
            result.fail(f"bash -n failed for {path}: {(proc.stderr or proc.stdout).strip()}")
    result.details["checked_files"] = checked
    return result


def parse_slurm_accounts(path: Path) -> list[str]:
    accounts: list[str] = []
    account_re = re.compile(r"^#SBATCH\s+--account(?:=|\s+)(\S+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = account_re.match(line.strip())
        if match:
            accounts.append(match.group(1))
    return accounts


def check_slurm_accounts(paths: list[Path], expected_account: str) -> CheckResult:
    result = CheckResult("slurm_accounts")
    accounts_by_file: dict[str, list[str]] = {}
    for path in paths:
        accounts = parse_slurm_accounts(path)
        accounts_by_file[str(path)] = accounts
        if not accounts:
            result.fail(f"Missing #SBATCH --account in {path}")
            continue
        for account in accounts:
            if account != expected_account:
                result.fail(f"{path}: expected --account={expected_account}, found {account}")
    result.details["expected_account"] = expected_account
    result.details["accounts_by_file"] = accounts_by_file
    return result



def check_dry_run_hooks(paths: list[Path]) -> CheckResult:
    result = CheckResult("dry_run_hooks")
    hooks_by_file: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hooks = []
        for marker in ("DRY_RUN", "PREFLIGHT_ONLY", "print_command"):
            if marker in text:
                hooks.append(marker)
        if "TRAIN_CMD=(" in text or "KBP_CMD=(" in text:
            hooks.append("command_array")
        hooks_by_file[str(path)] = hooks
        missing = {"DRY_RUN", "PREFLIGHT_ONLY", "print_command", "command_array"} - set(hooks)
        if missing:
            result.fail(f"{path}: missing dry-run hook markers: {sorted(missing)}")
    result.details["hooks_by_file"] = hooks_by_file
    return result

def scan_drift_patterns(paths: list[Path]) -> CheckResult:
    result = CheckResult("script_drift_scan")
    hits: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for name, pattern in DRIFT_PATTERNS.items():
            if pattern in text:
                hits.setdefault(str(path), []).append(name)
    if hits:
        result.warn(f"Potential stale path/account strings found: {hits}")
    result.details["hits"] = hits
    return result


def build_manifest(args: argparse.Namespace, checks: list[CheckResult]) -> dict[str, Any]:
    failures = sum(1 for item in checks if item.status == "fail")
    warnings = sum(1 for item in checks if item.status == "warn")
    return {
        "generated_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "status": "fail" if failures else "pass",
        "warning_check_count": warnings,
        "args": {
            "repo_root": str(args.repo_root),
            "project_root": str(args.project_root),
            "cpro_root": str(args.cpro_root),
            "schema_mode": args.schema_mode,
            "expected_account": args.expected_account,
        },
        "git": {
            "truthrl_head": _git_output(args.repo_root, ["rev-parse", "HEAD"]),
            "truthrl_branch": _git_output(args.repo_root, ["branch", "--show-current"]),
            "truthrl_status_short": _git_output(args.repo_root, ["status", "--short", "--branch"]),
        },
        "checks": [asdict(item) for item in checks],
    }


def _default_manifest_path(repo_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "data" / "medical" / "reports" / f"preflight_manifest_{stamp}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--cpro-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--rl-files", nargs="*", type=Path, default=None)
    parser.add_argument("--sft-files", nargs="*", type=Path, default=None)
    parser.add_argument("--schema-mode", choices=["skip", "existing", "strict"], default="existing")
    parser.add_argument("--script-glob", action="append", default=None)
    parser.add_argument("--slurm-glob", action="append", default=None)
    parser.add_argument("--dry-run-glob", action="append", default=None)
    parser.add_argument("--expected-account", default=DEFAULT_EXPECTED_ACCOUNT)
    parser.add_argument("--strict-ambient-import", action="store_true")
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--no-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.repo_root = _resolve(args.repo_root)
    args.project_root = _resolve(args.project_root) if args.project_root else args.repo_root.parent
    args.cpro_root = _resolve(args.cpro_root) if args.cpro_root else args.project_root / "CPRO"
    model_path = args.model_path or args.project_root / "models" / "stagec_c8_8b_len10240_ep15_4gpu_step17175" / "merged_model"
    model_path = _resolve(model_path)

    rl_files = [_resolve(_as_abs(args.repo_root, p)) for p in (args.rl_files or DEFAULT_RL_FILES)]
    sft_files = [_resolve(_as_abs(args.repo_root, p)) for p in (args.sft_files or DEFAULT_SFT_FILES)]
    script_globs = args.script_glob or DEFAULT_SHELL_GLOBS
    slurm_globs = args.slurm_glob or DEFAULT_SLURM_GLOBS
    dry_run_globs = args.dry_run_glob or DEFAULT_DRY_RUN_GLOBS
    shell_files = discover_files(args.repo_root, script_globs)
    slurm_files = discover_files(args.repo_root, slurm_globs)
    dry_run_files = discover_files(args.repo_root, dry_run_globs)
    syntax_files = shell_files + [p for p in slurm_files if p not in shell_files]

    checks = [
        check_paths(
            args.repo_root,
            args.project_root,
            args.cpro_root,
            model_path,
            rl_files,
            sft_files,
            strict_files=args.schema_mode == "strict",
        ),
        check_verl_import(args.repo_root, args.cpro_root, args.strict_ambient_import),
        check_schema(args.repo_root, rl_files, sft_files, args.schema_mode),
        check_bash_syntax(syntax_files),
        check_dry_run_hooks(dry_run_files),
        check_slurm_accounts(slurm_files, args.expected_account),
        scan_drift_patterns(syntax_files),
    ]
    manifest = build_manifest(args, checks)

    if not args.no_manifest:
        manifest_path = args.manifest_out or _default_manifest_path(args.repo_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] wrote preflight manifest: {manifest_path}")

    print(f"[{manifest['status'].upper()}] medical runtime preflight")
    for check in checks:
        print(f"  - {check.name}: {check.status}")
        for warning in check.warnings:
            print(f"    warning: {warning}")
        for error in check.errors:
            print(f"    error: {error}")

    return 1 if manifest["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
