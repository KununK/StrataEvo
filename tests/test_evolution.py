import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from strataevo.evolution.generation import _prepare_generation_dir, run_one_generation
from strataevo.evolution.mutator import _MetaAgent, _run_refinement_session
from strataevo.evolution.runtime.evaluation import promotion_decision
from strataevo.evolution.runtime.structured import StructuredResponseError
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Agent, Message, ScriptedModel, ToolCall, ToolRegistry, tool


class EvolutionTests(unittest.TestCase):
    def test_invalid_decision_rejects_generation_and_advances_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "tinyagent" / "agent.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (root / ".gitignore").write_text("evolution/\n", encoding="utf-8")
            self._git(root, "init")
            self._git(root, "config", "user.email", "test@example.com")
            self._git(root, "config", "user.name", "Test")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "parent")
            branch = self._git(root, "branch", "--show-current").strip()
            commit = self._git(root, "rev-parse", "HEAD").strip()

            run_dir = root / "evolution" / "runs" / "test"
            run_dir.mkdir(parents=True)
            config = EvolutionConfig(repo=str(root), run_name="test", branch=branch)
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            report = EvaluationReport(0.5, {"passed": 5}, "parent", "parent.log")
            state_path = run_dir / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "next_generation": 1,
                        "current_commit": commit,
                        "current_model": config.model,
                        "current_adapter": None,
                        "current_context": None,
                        "current_tool_profile": None,
                        "current_report": report.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            evaluator = Mock()

            with (
                patch(
                    "strataevo.evolution.generation.create_evaluator",
                    return_value=evaluator,
                ),
                patch(
                    "strataevo.evolution.generation.decide_evolution",
                    side_effect=StructuredResponseError("invalid decision"),
                ),
            ):
                result = run_one_generation(config_path)

            state = json.loads(state_path.read_text(encoding="utf-8"))
            record = json.loads(
                (run_dir / "generation-0001" / "record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result, 0)
            self.assertEqual(state["next_generation"], 2)
            self.assertEqual(state["current_report"], report.to_dict())
            self.assertEqual(record["layer"], "decision")
            self.assertEqual(record["decision"], "rejected")
            self.assertEqual(record["outcome"], "decision_invalid")
            self.assertFalse((run_dir / "generation-0001" / "failure.json").exists())

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

    @staticmethod
    def _git(root: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout


if __name__ == "__main__":
    unittest.main()
