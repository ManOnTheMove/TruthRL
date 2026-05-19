from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_PATH = REPO_ROOT / "training/verl/verl/utils/reward_score/clinical_medqa_reward.py"


def load_reward_module():
    spec = importlib.util.spec_from_file_location("clinical_medqa_reward_for_test", REWARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ground_truth(target="A"):
    return {
        "target": [target],
        "choices": [
            "A: first",
            "B: second",
            "C: third",
            "D: fourth",
        ],
    }


class TestClinicalMedqaRewardCompat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reward = load_reward_module()

    def test_training_reward_keeps_strict_single_boxed_policy(self):
        result = self.reward.compute_score("medical", "\\boxed{A} then \\boxed{B}", ground_truth("A"))
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["prediction_type"], 0)
        self.assertEqual(result["is_boxed_valid"], 0)

    def test_training_reward_accepts_single_correct_boxed_choice(self):
        result = self.reward.compute_score("medical", "<answer>\\boxed{A}</answer>", ground_truth("A"))
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["prediction_type"], 3)
        self.assertEqual(result["is_boxed_valid"], 1)

    def test_normalize_choice_does_not_turn_aspirin_into_option_a(self):
        self.assertEqual(self.reward.normalize_choice("Aspirin"), "")
        result = self.reward.compute_score("medical", "\\boxed{Aspirin}", ground_truth("A"))
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["prediction_type"], 2)
        self.assertEqual(result["is_boxed_valid"], 0)

    def test_reward_module_exports_parser_policies_for_scripts(self):
        self.assertEqual(self.reward.POLICY_SINGLE_BOXED_STRICT, "single_boxed_strict")
        self.assertEqual(self.reward.POLICY_FINAL_ANSWER_LAST_BOXED, "final_answer_last_boxed")
        self.assertEqual(self.reward.POLICY_FIRST_BOXED_LEGACY, "first_boxed_legacy")


if __name__ == "__main__":
    unittest.main()
