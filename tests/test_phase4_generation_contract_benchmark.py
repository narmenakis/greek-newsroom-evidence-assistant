import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.evaluation.generation_contract_benchmark import (
    render_markdown,
    run_contract_benchmark,
)


class Phase4GenerationContractBenchmarkTests(unittest.TestCase):
    def test_report_contains_all_passing_mocked_cases(self):
        report = run_contract_benchmark()
        self.assertEqual(report["aggregate"]["case_count"], 9)
        self.assertEqual(report["aggregate"]["passed_count"], 9)
        self.assertEqual(report["aggregate"]["failed_count"], 0)
        self.assertEqual(report["external_provider_calls"], 0)
        self.assertEqual(report["provider_mode"], "mocked")

    def test_markdown_report_is_concise_and_honest_about_scope(self):
        markdown = render_markdown(run_contract_benchmark())
        self.assertIn("9/9 cases passed", markdown)
        self.assertIn("provider calls were made", markdown)
        self.assertIn("not evaluate semantic", markdown)
        self.assertIn("provider_http_failures_are_typed", markdown)


if __name__ == "__main__":
    unittest.main()
