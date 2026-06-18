from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/medical/kbp/extract_kbp_metrics.py"


def load_script_module():
    if importlib.util.find_spec("pandas") is None:
        raise unittest.SkipTest("pandas is not available")
    if importlib.util.find_spec("pyarrow") is None:
        raise unittest.SkipTest("pyarrow is not available")

    spec = importlib.util.spec_from_file_location("kbp_extract_metrics_for_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestKbpExtractMetricsPolicy(unittest.TestCase):
    def test_static_policy_markers_are_present_without_optional_deps(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('PARSER_POLICY = "final_answer_last_boxed"', text)
        self.assertIn('METRIC_NAMESPACE = "new_parser_final_answer_last_boxed"', text)
        self.assertIn("KBP_results/scripts/extract_kbp_metrics.py", text)
        self.assertIn('"code_commit": code_commit', text)
        self.assertIn('"parser_policy": PARSER_POLICY', text)
        self.assertIn('"new_parser_correct_count"', text)
        self.assertIn('"new_parser_difficulty_bucket"', text)

    def test_bucket_metrics_use_new_parser_column(self):
        script = load_script_module()
        pd = script.pd
        question_df = pd.DataFrame(
            [
                {"question_id": "q1", "new_parser_difficulty_bucket": "0"},
                {"question_id": "q2", "new_parser_difficulty_bucket": "1-63"},
                {"question_id": "q3", "new_parser_difficulty_bucket": "1-63"},
            ]
        )

        bucket_df = script.build_bucket_metrics(question_df)

        self.assertIn("new_parser_difficulty_bucket", bucket_df.columns)
        self.assertNotIn("difficulty_bucket", bucket_df.columns)
        self.assertTrue((bucket_df["parser_policy"] == "final_answer_last_boxed").all())
        self.assertTrue((bucket_df["metric_namespace"] == "new_parser_final_answer_last_boxed").all())


if __name__ == "__main__":
    unittest.main()
