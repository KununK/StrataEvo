import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.bfcl.official import format_call, locate_bfcl_root
from eval.bfcl.run import run_agent_task
from tinyagent import Message, ScriptedModel, Tool, ToolCall


class BFCLTests(unittest.TestCase):
    def test_locates_direct_and_gorilla_checkout_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "berkeley-function-call-leaderboard" / "bfcl_eval"
            data = package / "data"
            data.mkdir(parents=True)
            (data / "BFCL_v4_multi_turn_base.json").write_text("{}\n", encoding="utf-8")

            self.assertEqual(locate_bfcl_root(root), package)
            self.assertEqual(locate_bfcl_root(package), package)

    def test_formats_function_calls_for_official_checker(self):
        self.assertEqual(
            format_call("search", {"query": "agent", "limit": 3}),
            "search(query='agent', limit=3)",
        )

    def test_runs_multi_turn_task_and_preserves_official_call_shape(self):
        observed = []

        class FakeOfficial:
            def __init__(self, _root=None):
                pass

            def create_tools(self, _task, _addenda):
                return [
                    Tool(
                        "lookup",
                        "Look up a value.",
                        lambda value: {"value": value},
                        {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                        },
                    )
                ]

            def check(self, _task, calls):
                observed.extend(calls)
                return {"valid": True}

        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    tool_calls=[ToolCall("call-1", "lookup", {"value": 3})],
                ),
                Message("assistant", "Done."),
                Message(
                    "assistant",
                    tool_calls=[ToolCall("call-2", "lookup", {"value": 4})],
                ),
                Message("assistant", "Done again."),
            ]
        )
        task = {
            "task_id": "multi_turn_base_test",
            "prompt": "Look up 3, then 4.",
            "question": [
                [{"role": "user", "content": "Look up 3."}],
                [{"role": "user", "content": "Now look up 4."}],
            ],
            "_bfcl_root": "/unused",
        }

        with tempfile.TemporaryDirectory() as directory, patch(
            "eval.bfcl.run.OfficialBFCL", FakeOfficial
        ):
            generation, result = run_agent_task(
                task,
                model,
                Path(directory) / "unused.py",
            )

        self.assertTrue(result["passed"])
        self.assertEqual(observed, [[['lookup(value=3)']], [['lookup(value=4)']]])
        self.assertEqual(generation["function_calls"], observed)

if __name__ == "__main__":
    unittest.main()
