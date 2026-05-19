from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "scripts" / "medical" / "preflight_medical_runtime.py"

spec = importlib.util.spec_from_file_location("preflight_medical_runtime", PREFLIGHT_PATH)
assert spec is not None
preflight = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = preflight
spec.loader.exec_module(preflight)


class MedicalPreflightTest(unittest.TestCase):
    def test_parse_slurm_accounts_supports_equal_and_space_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.slurm"
            path.write_text(
                "#!/bin/bash\n"
                "#SBATCH --account=def-zshakeri\n"
                "#SBATCH --account def-zshakeri\n",
                encoding="utf-8",
            )

            self.assertEqual(preflight.parse_slurm_accounts(path), ["def-zshakeri", "def-zshakeri"])

    def test_slurm_account_check_fails_on_wrong_account(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.slurm"
            path.write_text("#!/bin/bash\n#SBATCH --account=def-zshakeri_gpu\n", encoding="utf-8")

            result = preflight.check_slurm_accounts([path], "def-zshakeri")

            self.assertEqual(result.status, "fail")
            self.assertIn("def-zshakeri_gpu", result.errors[0])

    def test_bash_syntax_check_uses_static_parse_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.sh"
            marker = Path(tmp) / "marker"
            path.write_text(f"#!/usr/bin/env bash\ntouch {marker}\n", encoding="utf-8")

            result = preflight.check_bash_syntax([path])

            self.assertEqual(result.status, "pass")
            self.assertFalse(marker.exists())

    def test_dry_run_hook_check_requires_guard_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.sh"
            missing.write_text("#!/usr/bin/env bash\necho run\n", encoding="utf-8")

            result = preflight.check_dry_run_hooks([missing])

            self.assertEqual(result.status, "fail")
            self.assertIn("DRY_RUN", result.errors[0])

    def test_dry_run_hook_check_passes_for_command_array_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guarded = Path(tmp) / "guarded.sh"
            guarded.write_text(
                "#!/usr/bin/env bash\n"
                "DRY_RUN=\"${DRY_RUN:-0}\"\n"
                "PREFLIGHT_ONLY=\"${PREFLIGHT_ONLY:-0}\"\n"
                "print_command() { printf x; }\n"
                "TRAIN_CMD=(python3 -m example)\n",
                encoding="utf-8",
            )

            result = preflight.check_dry_run_hooks([guarded])

            self.assertEqual(result.status, "pass")
            self.assertIn("command_array", result.details["hooks_by_file"][str(guarded)])

    def test_expected_verl_spec_resolves_to_truthrl_vendor_tree(self) -> None:
        result = preflight.check_verl_import(REPO_ROOT, REPO_ROOT.parent / "CPRO", strict_ambient=False)

        self.assertNotEqual(result.status, "fail")
        self.assertIn("/training/verl/verl/", result.details["expected_spec_origin"])


if __name__ == "__main__":
    unittest.main()
