import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.generation import _prepare_generation_dir
from strataevo.evolution.mutator import _MetaAgent, _run_refinement_session
from strataevo.evolution.runtime.evaluation import promotion_decision
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Agent, Message, ScriptedModel, ToolCall, ToolRegistry, tool


class EvolutionTests(unittest.TestCase):
    def test_meta_agent_loop_is_independent_from_evolvable_agent_loop(self):
        @tool
        def inspect() -> str:
            """Inspect state."""
            return "observed"

        model = ScriptedModel(
            [
                Message("assistant", tool_calls=[ToolCall("1", "inspect", {})]),
                Message("assistant", "fixed"),
            ]
        )
        agent = _MetaAgent(model, ToolRegistry([inspect]), "system", 2, 10_000)

        result = agent.run("repair")

        self.assertEqual(result.output, "fixed")
        self.assertEqual(result.steps, 2)
        self.assertEqual(
            [message.role for message in model.requests[1][-3:]],
            ["user", "assistant", "tool"],
        )

    def test_failed_generation_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation-0001"
            path.mkdir()
            (path / "failure.json").write_text("{}", encoding="utf-8")
            _prepare_generation_dir(path)
            self.assertTrue(path.is_dir())
            self.assertTrue(path.with_name("generation-0001-failed-0001").is_dir())

    def test_promotion_requires_minimum_pass_gain(self):
        parent = EvaluationReport(0.5, {"passed": 100}, "p", "p.log")
        equal = EvaluationReport(0.5, {"passed": 100}, "c", "c.log")
        better = EvaluationReport(0.51, {"passed": 101}, "c", "c.log")
        self.assertFalse(promotion_decision(parent, equal)[0])
        self.assertTrue(promotion_decision(parent, better)[0])

    def test_refinement_passes_prior_messages_without_disk_session(self):
        model = ScriptedModel([Message("assistant", "continue"), Message("assistant", "FINALIZE")])
        feedback = iter(
            [
                '{"outcome":"evaluated","candidate_task_score":0.5,"evaluations_remaining":1}',
                '{"outcome":"evaluated","candidate_task_score":0.5,"evaluations_remaining":1}',
            ]
        )
        result = _run_refinement_session(
            Agent(model),
            "initial task",
            lambda: next(feedback),
            EvolutionConfig(repo=".", run_name="test", mutator_rounds=2),
        )

        second_request = [message.content for message in model.requests[1]]
        self.assertEqual(second_request[-3:-1], ["initial task", "continue"])
        self.assertIn("Evaluation feedback", second_request[-1])
        self.assertEqual(result.steps, 2)


if __name__ == "__main__":
    unittest.main()
