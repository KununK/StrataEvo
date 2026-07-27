import tempfile
import unittest
from pathlib import Path

from tinyagent import (
    Agent,
    Message,
    ModelResponse,
    ScriptedModel,
    SessionStore,
    ToolCall,
    ToolRegistry,
    Usage,
    Workspace,
    tool,
)


class AgentTests(unittest.TestCase):
    def test_agent_executes_tool_and_returns_answer(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        model = ScriptedModel(
            [
                ModelResponse(
                    Message(
                        "assistant",
                        tool_calls=[ToolCall("1", "add", {"a": 2, "b": 3})],
                    ),
                    Usage(10, 2),
                ),
                ModelResponse(Message("assistant", "five"), Usage(8, 1)),
            ]
        )
        events = []
        agent = Agent(model, ToolRegistry([add]))
        agent.events.subscribe(lambda event: events.append(event.type))

        result = agent.run("calculate")

        self.assertEqual(result.output, "five")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.usage, Usage(18, 3))
        self.assertEqual(result.messages[-2].role, "tool")
        self.assertEqual(result.messages[-2].content, "5")
        self.assertEqual(
            events,
            [
                "run_start",
                "model_start",
                "model_end",
                "tool_start",
                "tool_end",
                "model_start",
                "model_end",
                "run_end",
            ],
        )

    def test_unknown_tool_error_is_returned_to_model(self):
        model = ScriptedModel(
            [
                Message("assistant", tool_calls=[ToolCall("1", "missing", {})]),
                Message("assistant", "recovered"),
            ]
        )
        result = Agent(model).run("go")
        self.assertIn("unknown tool", result.messages[-2].content)
        self.assertEqual(result.output, "recovered")

    def test_approval_can_deny_side_effect(self):
        called = False

        @tool(requires_approval=True)
        def dangerous() -> str:
            """Do something."""
            nonlocal called
            called = True
            return "done"

        model = ScriptedModel(
            [
                Message("assistant", tool_calls=[ToolCall("1", "dangerous", {})]),
                Message("assistant", "denied"),
            ]
        )
        result = Agent(model, ToolRegistry([dangerous]), approval=lambda _n, _a: False).run("go")
        self.assertFalse(called)
        self.assertIn("denied by approval policy", result.messages[-2].content)

    def test_session_is_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            first = Agent(ScriptedModel([Message("assistant", "one")]), session_store=store)
            first.run("hello", session_id="paper-1")

            second_model = ScriptedModel([Message("assistant", "two")])
            Agent(second_model, session_store=store).run("continue", session_id="paper-1")
            sent = second_model.requests[0]
            self.assertEqual(
                [message.content for message in sent][-3:], ["hello", "one", "continue"]
            )

    def test_max_steps_stops_loop(self):
        calls = [
            Message("assistant", tool_calls=[ToolCall(str(i), "missing", {})]) for i in range(2)
        ]
        result = Agent(ScriptedModel(calls), max_steps=2).run("loop")
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.steps, 2)

    def test_workspace_tools_are_rooted_and_write_requires_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            tools = ToolRegistry(workspace.tools())
            write = tools.get("write_file")
            read = tools.get("read_file")
            self.assertIsNotNone(write)
            self.assertTrue(write.requires_approval)
            self.assertIsNotNone(read)
            write.run({"path": "notes/a.txt", "content": "alpha\nbeta"})
            self.assertIn(
                "alpha", read.run({"path": "notes/a.txt", "start_line": 1, "end_line": 1})
            )
            with self.assertRaises(PermissionError):
                read.run({"path": "../outside.txt"})

    def test_workspace_limits_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "long.txt")
            path.write_text("x" * 100, encoding="utf-8")
            read = ToolRegistry(Workspace(directory, max_output=10).tools()).get("read_file")
            output = read.run({"path": "long.txt"})
            self.assertTrue(output.endswith("... output truncated"))

    def test_compaction_preserves_latest_complete_tool_turn(self):
        history = [
            Message("system", "system"),
            Message("user", "old " * 100),
            Message("assistant", "old answer"),
        ]
        model = ScriptedModel(
            [
                Message("assistant", tool_calls=[ToolCall("1", "missing", {})]),
                Message("assistant", "done"),
            ]
        )
        agent = Agent(model, context_limit_chars=50)
        result = agent.run("new task", history=history)
        second_request = model.requests[1]
        self.assertEqual(
            [message.role for message in second_request[-3:]], ["user", "assistant", "tool"]
        )
        self.assertEqual(second_request[-2].tool_calls[0].id, second_request[-1].tool_call_id)
        self.assertEqual(result.output, "done")

    def test_compaction_counts_tool_arguments_in_single_user_loop(self):
        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    tool_calls=[ToolCall("old", "missing", {"content": "x" * 600})],
                ),
                Message(
                    "assistant",
                    tool_calls=[ToolCall("new", "missing", {"content": "y" * 600})],
                ),
                Message("assistant", "done"),
            ]
        )
        agent = Agent(model, context_limit_chars=900)

        result = agent.run("one long tool-driven task")

        third_request = model.requests[2]
        call_ids = [call.id for message in third_request for call in message.tool_calls]
        self.assertNotIn("old", call_ids)
        self.assertIn("new", call_ids)
        self.assertEqual([message.role for message in third_request[-2:]], ["assistant", "tool"])
        self.assertEqual(third_request[-2].tool_calls[0].id, third_request[-1].tool_call_id)
        self.assertEqual(result.output, "done")

    def test_agent_rejects_non_positive_limits(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedModel([]), max_steps=0)


if __name__ == "__main__":
    unittest.main()
