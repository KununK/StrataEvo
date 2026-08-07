import json
import unittest

from strataevo.evolution.decision import EvolutionDecider, EvolutionDecision, EvolutionLayer
from strataevo.evolution.evidence import EvidenceBundle, TaskEvidence, ToolEvent
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Message, ScriptedModel


class DecisionTests(unittest.TestCase):
    def test_decision_uses_ordered_events_and_names_mutable_files(self):
        response = {
            "layer": "architecture",
            "evidence": ["task/1 removed its output after writing it"],
            "affected_tasks": ["task/1"],
            "hypothesis": "The loop permits destructive cleanup after producing the output.",
            "intervention": "Stop after the verified output is produced.",
            "likely_files": ["src/tinyagent/agent.py"],
        }
        model = ScriptedModel([Message("assistant", json.dumps(response))])
        decision = EvolutionDecider(model).decide(
            self._bundle(),
            EvaluationReport(0.5, {}, "evaluation", "evaluation.log"),
            [],
            EvolutionConfig(repo=".", run_name="test"),
        )
        self.assertEqual(decision.layer, EvolutionLayer.ARCHITECTURE)
        system_prompt = model.requests[0][0].content
        request = model.requests[0][1].content
        self.assertIn("executor can change the observed cause", system_prompt)
        self.assertIn("generated-code correctness", system_prompt)
        self.assertIn("operations inside generated code are not agent", system_prompt)
        self.assertIn("assertion failure is model evidence", system_prompt)
        self.assertIn("later observed tool call", system_prompt)
        self.assertIn("rather than inventing a subsystem", system_prompt)
        self.assertIn("Current: ...; Change: ...; Verify: ...", system_prompt)
        self.assertLess(request.index("write_file"), request.index("run_shell"))
        self.assertIn("src/tinyagent/agent.py", request)

    def test_decision_rejects_unknown_tasks(self):
        data = {
            "layer": "context",
            "evidence": ["x"],
            "affected_tasks": ["other"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": [],
        }
        with self.assertRaisesRegex(ValueError, "current evidence"):
            EvolutionDecision.from_dict(
                data, {"task/1"}, EvolutionConfig(repo=".", run_name="test")
            )

    def test_architecture_decision_cannot_escape_mutable_source(self):
        data = {
            "layer": "architecture",
            "evidence": ["x"],
            "affected_tasks": ["task/1"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": ["eval/run.py"],
        }
        with self.assertRaisesRegex(ValueError, "mutable likely_files"):
            EvolutionDecision.from_dict(
                data, {"task/1"}, EvolutionConfig(repo=".", run_name="test")
            )

    def test_architecture_decision_requires_an_existing_source_file(self):
        data = {
            "layer": "architecture",
            "evidence": ["x"],
            "affected_tasks": ["task/1"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": ["src/tinyagent/missing.py"],
        }
        with self.assertRaisesRegex(ValueError, "existing mutable likely_files"):
            EvolutionDecision.from_dict(
                data, {"task/1"}, EvolutionConfig(repo=".", run_name="test")
            )

    def test_external_decision_ignores_source_file_suggestions(self):
        data = {
            "layer": "model",
            "evidence": ["x"],
            "affected_tasks": ["task/1"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": ["src/tinyagent/agent.py"],
        }
        decision = EvolutionDecision.from_dict(
            data,
            {"task/1"},
            EvolutionConfig(
                repo=".", run_name="test", model_evolution=True, force_layer="model"
            ),
        )
        self.assertEqual(decision.likely_files, [])

    @staticmethod
    def _bundle():
        case = TaskEvidence(
            "task/1",
            "missing_candidate",
            False,
            "completed",
            3,
            False,
            True,
            [
                ToolEvent(1, "write_file", {"path": "solution.py"}, "wrote"),
                ToolEvent(2, "run_shell", {"command": "rm solution.py"}, "ok"),
            ],
            "missing",
            ["artifact_missing"],
        )
        return EvidenceBundle({"passed": 0}, {"artifact_missing": 1}, [case])


if __name__ == "__main__":
    unittest.main()
