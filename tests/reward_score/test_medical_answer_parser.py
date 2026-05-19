from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PARSER_PATH = REPO_ROOT / "training/verl/verl/utils/reward_score/medical_answer_parser.py"


def load_parser_module():
    spec = importlib.util.spec_from_file_location("medical_answer_parser_for_test", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestMedicalAnswerParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = load_parser_module()

    def test_strict_single_boxed_accepts_exactly_one_boxed(self):
        parsed = self.parser.parse_medical_answer(
            "Reasoning. \\boxed{A}",
            policy=self.parser.POLICY_SINGLE_BOXED_STRICT,
            choice_set={"A", "B", "C", "D"},
        )
        self.assertEqual(parsed.parse_status, "ok_choice")
        self.assertEqual(parsed.normalized_choice, "A")
        self.assertEqual(parsed.boxed_count, 1)

    def test_strict_single_boxed_rejects_no_or_multiple_boxed(self):
        no_boxed = self.parser.parse_medical_answer(
            "Reasoning only.",
            policy=self.parser.POLICY_SINGLE_BOXED_STRICT,
        )
        multi_boxed = self.parser.parse_medical_answer(
            "\\boxed{A} then \\boxed{B}",
            policy=self.parser.POLICY_SINGLE_BOXED_STRICT,
        )
        self.assertEqual(no_boxed.parse_status, "no_boxed")
        self.assertEqual(multi_boxed.parse_status, "multi_boxed")
        self.assertEqual(multi_boxed.boxed_count, 2)

    def test_final_answer_last_boxed_prefers_final_answer_block(self):
        parsed = self.parser.parse_medical_answer(
            "<think>try \\boxed{B}</think><answer>final \\boxed{C}</answer>",
            policy=self.parser.POLICY_FINAL_ANSWER_LAST_BOXED,
            choice_set={"A", "B", "C", "D"},
        )
        self.assertEqual(parsed.parse_status, "ok_choice")
        self.assertEqual(parsed.normalized_choice, "C")
        self.assertTrue(parsed.used_answer_block)
        self.assertEqual(parsed.boxed_count, 2)

    def test_final_answer_last_boxed_falls_back_to_global_last(self):
        parsed = self.parser.parse_medical_answer(
            "<answer>no box here</answer> earlier \\boxed{A} later \\boxed{D}",
            policy=self.parser.POLICY_FINAL_ANSWER_LAST_BOXED,
            choice_set={"A", "B", "C", "D"},
        )
        self.assertEqual(parsed.parse_status, "ok_choice")
        self.assertEqual(parsed.normalized_choice, "D")
        self.assertFalse(parsed.used_answer_block)

    def test_first_boxed_legacy_uses_first_global_boxed(self):
        parsed = self.parser.parse_medical_answer(
            "\\boxed{B} then \\boxed{A}",
            policy=self.parser.POLICY_FIRST_BOXED_LEGACY,
            choice_set={"A", "B", "C", "D"},
        )
        self.assertEqual(parsed.parse_status, "ok_choice")
        self.assertEqual(parsed.normalized_choice, "B")

    def test_abstain_variants_are_separate_from_choice_labels(self):
        parsed = self.parser.parse_medical_answer(
            "\\boxed{I don't know}",
            policy=self.parser.POLICY_SINGLE_BOXED_STRICT,
        )
        self.assertEqual(parsed.parse_status, "ok_abstain")
        self.assertEqual(parsed.normalized_choice, "I don't know")
        self.assertTrue(parsed.is_abstain)

    def test_normalize_choice_does_not_treat_medical_words_as_labels(self):
        self.assertEqual(self.parser.normalize_choice("A"), "A")
        self.assertEqual(self.parser.normalize_choice("A)"), "A")
        self.assertEqual(self.parser.normalize_choice("A."), "A")
        self.assertEqual(self.parser.normalize_choice("A: Aspirin"), "A")
        self.assertEqual(self.parser.normalize_choice("A - Aspirin"), "A")
        self.assertEqual(self.parser.normalize_choice("Aspirin"), "")

    def test_choice_out_of_set_is_explicit(self):
        parsed = self.parser.parse_medical_answer(
            "\\boxed{Z}",
            policy=self.parser.POLICY_SINGLE_BOXED_STRICT,
            choice_set={"A", "B", "C", "D"},
        )
        self.assertEqual(parsed.parse_status, "choice_out_of_set")
        self.assertEqual(parsed.normalized_choice, "Z")
        self.assertFalse(parsed.is_valid_choice)

    def test_balanced_boxed_parser_handles_nested_braces(self):
        self.assertEqual(self.parser.find_boxed_answers(r"\boxed{\text{A}}"), [r"\text{A}"])


if __name__ == "__main__":
    unittest.main()
