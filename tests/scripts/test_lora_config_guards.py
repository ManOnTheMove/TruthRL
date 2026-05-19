from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestLoraConfigGuards(unittest.TestCase):
    def test_merge_lora_validates_alpha_before_model_load(self):
        merge_lora = _load_module("scripts/medical/merge_lora.py", "merge_lora_for_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 32}), encoding="utf-8"
            )

            data = merge_lora._load_and_validate_adapter_config(str(adapter_dir), expected_lora_alpha=32)

        self.assertEqual(data["lora_alpha"], 32)

    def test_merge_lora_rejects_missing_or_nonpositive_alpha(self):
        merge_lora = _load_module("scripts/medical/merge_lora.py", "merge_lora_for_bad_alpha_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing lora_alpha"):
                merge_lora._load_and_validate_adapter_config(str(adapter_dir))

            (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be > 0"):
                merge_lora._load_and_validate_adapter_config(str(adapter_dir))

    def test_merge_lora_rejects_unexpected_alpha(self):
        merge_lora = _load_module("scripts/medical/merge_lora.py", "merge_lora_for_mismatch_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 64}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "lora_alpha mismatch"):
                merge_lora._load_and_validate_adapter_config(str(adapter_dir), expected_lora_alpha=32)

    def test_repair_lora_adapter_config_preserves_positive_alpha(self):
        repair = _load_module("scripts/medical/repair_lora_adapter_config.py", "repair_lora_for_preserve_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_config = Path(tmpdir) / "adapter_config.json"
            adapter_config.write_text(json.dumps({"lora_alpha": 64}), encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                [
                    "repair_lora_adapter_config.py",
                    "--adapter-config",
                    str(adapter_config),
                    "--expected-lora-alpha",
                    "32",
                ],
            ):
                repair.main()

            self.assertEqual(json.loads(adapter_config.read_text(encoding="utf-8"))["lora_alpha"], 64)

    def test_model_mergers_require_explicit_lora_alpha_for_adapter_export(self):
        paths = [
            REPO_ROOT / "training/verl/verl/model_merger/base_model_merger.py",
            REPO_ROOT / "training/verl/scripts/legacy_model_merger.py",
        ]

        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("--lora-alpha", source)
            self.assertIn("LoRA adapter export requires --lora-alpha", source)
            self.assertNotIn("\"lora_alpha\": 0", source)


if __name__ == "__main__":
    unittest.main()
