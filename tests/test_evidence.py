import json
import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.evidence import (
    CodingAgentEvidenceCollector,
    EvidenceBundle,
    TaskEvidence,
)


class EvidenceCollectorTests(unittest.TestCase):
    def test_collects_artifact_lifecycle_and_persists_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)

            bundle = CodingAgentEvidenceCollector("humaneval").collect_and_write(root)

            self.assertEqual(
                [case.task_id for case in bundle.cases], ["HumanEval/1", "HumanEval/2"]
            )
            passed, missing = bundle.cases
            self.assertTrue(passed.candidate_present)
            self.assertIn("max_steps", passed.signals)
            self.assertTrue(missing.candidate_created)
            self.assertEqual(
                missing.signals,
                [
                    "missing_candidate",
                    "artifact_missing",
                    "artifact_created_then_missing",
                    "completed_without_artifact",
                ],
            )
            self.assertEqual(bundle.signal_counts["artifact_created_then_missing"], 1)
            evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["cases"][1]["task_id"], "HumanEval/2")
            self.assertNotIn("candidate_deleted", evidence["cases"][1])

    def test_legacy_delete_signals_are_ignored(self):
        data = {
            "task_id": "HumanEval/0",
            "entry_point": "answer",
            "status": "missing_candidate",
            "passed": False,
            "stop_reason": "completed",
            "steps": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "generation_seconds": 0.1,
            "test_seconds": 0.0,
            "candidate_path": None,
            "candidate_present": False,
            "candidate_created": True,
            "candidate_deleted": True,
            "tool_sequence": ["run_shell"],
            "shell_commands": ["rm solution.py"],
            "error": "",
            "generation_path": "generations.jsonl",
            "session_path": None,
            "signals": ["artifact_deleted"],
        }

        case = TaskEvidence.from_dict(data)

        self.assertFalse(hasattr(case, "artifact_delete_attempted"))
        self.assertNotIn("artifact_deleted", case.signals)

        bundle = EvidenceBundle.from_dict(
            {
                "evaluator": "humaneval",
                "source_dir": "evaluation",
                "summary": {},
                "signal_counts": {"artifact_deleted": 1},
                "cases": [data],
            }
        )
        self.assertEqual(bundle.signal_counts, {})

    def test_rejects_results_without_matching_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            (root / "generations.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "task mismatch"):
                CodingAgentEvidenceCollector("humaneval").collect(root)

    def test_generic_collector_records_active_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)

            bundle = CodingAgentEvidenceCollector("mbpp").collect(root)

            self.assertEqual(bundle.evaluator, "mbpp")

    @staticmethod
    def _write_fixture(root: Path) -> None:
        (root / "candidates").mkdir(parents=True)
        (root / "sessions").mkdir()
        (root / "candidates/HumanEval_1.py").write_text(
            "def answer(): return 1\n", encoding="utf-8"
        )
        (root / "sessions/HumanEval_2.json").write_text('{"messages": []}\n', encoding="utf-8")
        (root / "summary.json").write_text(
            json.dumps({"evaluated": 2, "passed": 1, "pass_at_1": 0.5}), encoding="utf-8"
        )
        results = [
            {
                "task_id": "HumanEval/1",
                "entry_point": "answer",
                "passed": True,
                "status": "pass",
                "test_seconds": 0.1,
                "stderr": "",
            },
            {
                "task_id": "HumanEval/2",
                "entry_point": "missing",
                "passed": False,
                "status": "missing_candidate",
                "test_seconds": 0.0,
                "stderr": "solution.py was not created",
            },
        ]
        generations = [
            {
                "task_id": "HumanEval/1",
                "stop_reason": "max_steps",
                "steps": 12,
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "generation_seconds": 1.0,
                "messages": [],
            },
            {
                "task_id": "HumanEval/2",
                "stop_reason": "completed",
                "steps": 4,
                "usage": {"input_tokens": 12, "output_tokens": 3},
                "generation_seconds": 2.0,
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "name": "write_file",
                                "arguments": {"path": "solution.py", "content": "pass\n"},
                            },
                            {
                                "name": "run_shell",
                                "arguments": {"command": "rm -f solution.py"},
                            },
                        ],
                    }
                ],
            },
        ]
        EvidenceCollectorTests._write_jsonl(root / "results.jsonl", results)
        EvidenceCollectorTests._write_jsonl(root / "generations.jsonl", generations)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
