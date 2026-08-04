import tempfile
import unittest
from pathlib import Path

from eval.humaneval.execution import evaluate_source
from eval.humaneval.run import run_agent_task
from tinyagent import Message, ScriptedModel, ToolCall

TASK = {
    "task_id": "HumanEval/0",
    "prompt": 'def add(a, b):\n    """Return the sum."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
}


class HumanEvalTests(unittest.TestCase):
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
            generation, result = run_agent_task(TASK, model, candidate)
            self.assertTrue(result["passed"])
            self.assertEqual(result["agent_steps"], 2)
            self.assertEqual(candidate.read_text(encoding="utf-8"), source)
            self.assertEqual(generation["stop_reason"], "completed")


if __name__ == "__main__":
    unittest.main()
