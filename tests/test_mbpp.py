import argparse
import json
import tempfile
import unittest
from pathlib import Path

from eval.mbpp.execution import evaluate_source
from eval.mbpp.run import load_tasks, render_task_file, run_agent_task
from tinyagent import Message, ScriptedModel, ToolCall

TASK = {
    "task_id": "11",
    "prompt": "Write a function to add two integers.",
    "entry_point": "add",
    "test_list": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
    "test_setup": "",
    "test_imports": "",
}


class MBPPTests(unittest.TestCase):
    def test_evaluate_source_pass_and_failure(self):
        passed = evaluate_source(TASK, "def add(a, b):\n    return a + b\n")
        failed = evaluate_source(TASK, "def add(a, b):\n    return a - b\n")

        self.assertTrue(passed["passed"])
        self.assertEqual(passed["status"], "pass")
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["status"], "assertion_error")

    def test_agent_creates_and_passes_candidate(self):
        source = "def add(a, b):\n    return a + b\n"
        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    tool_calls=[
                        ToolCall(
                            "write-1",
                            "write_file",
                            {"path": "solution.py", "content": source},
                        )
                    ],
                ),
                Message("assistant", "solution.py is complete."),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            sessions = Path(directory) / "sessions"
            generation, result = run_agent_task(TASK, model, candidate, session_dir=sessions)

            self.assertTrue(result["passed"])
            self.assertEqual(generation["session_id"], "11")
            self.assertEqual(candidate.read_text(encoding="utf-8"), source)
            self.assertTrue((sessions / "11.json").is_file())

    def test_local_dataset_is_normalized(self):
        row = {
            "task_id": 11,
            "text": "Write a function to add two integers.",
            "test_list": ["assert add(2, 3) == 5"],
            "test_imports": ["import math"],
            "test_setup_code": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tasks.json"
            dataset.write_text(json.dumps([row]), encoding="utf-8")
            args = argparse.Namespace(
                dataset_file=str(dataset),
                dataset="unused",
                dataset_config="sanitized",
                split="test",
                offset=0,
                limit=None,
            )

            tasks = load_tasks(args)

        self.assertEqual(tasks[0]["task_id"], "11")
        self.assertEqual(tasks[0]["entry_point"], "add")
        self.assertIn("TESTS =", render_task_file(tasks[0]))


if __name__ == "__main__":
    unittest.main()
