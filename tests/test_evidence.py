import json
import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.evidence import CodingAgentEvidenceCollector


class EvidenceTests(unittest.TestCase):
    def test_collector_preserves_ordered_tool_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text('{"evaluated": 1}', encoding="utf-8")
            (root / "results.jsonl").write_text(
                json.dumps({"task_id": "task/1", "status": "missing_candidate", "passed": False})
                + "\n",
                encoding="utf-8",
            )
            messages = [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "write_file",
                            "arguments": {"path": "solution.py"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "1", "content": "wrote"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "2",
                            "name": "run_shell",
                            "arguments": {"command": "rm solution.py"},
                        }
                    ],
                },
            ]
            (root / "generations.jsonl").write_text(
                json.dumps({"task_id": "task/1", "messages": messages, "stop_reason": "completed"})
                + "\n",
                encoding="utf-8",
            )

            bundle = CodingAgentEvidenceCollector().collect_and_write(root)

            self.assertEqual(
                [event.name for event in bundle.cases[0].tool_events], ["write_file", "run_shell"]
            )
            self.assertIn("artifact_created_then_missing", bundle.cases[0].signals)
            self.assertTrue((root / "evidence.json").is_file())

    def test_collector_does_not_require_artifacts_for_trajectory_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text('{"evaluated": 1}', encoding="utf-8")
            (root / "results.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "task/1",
                        "status": "incorrect_calls",
                        "passed": False,
                        "artifact_expected": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "generations.jsonl").write_text(
                json.dumps({"task_id": "task/1", "messages": []}) + "\n",
                encoding="utf-8",
            )

            bundle = CodingAgentEvidenceCollector().collect_and_write(root)

            self.assertEqual(bundle.cases[0].signals, ["incorrect_calls"])


if __name__ == "__main__":
    unittest.main()
