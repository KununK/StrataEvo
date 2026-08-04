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
        request = model.requests[0][1].content
        self.assertLess(request.index("write_file"), request.index("run_shell"))

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
