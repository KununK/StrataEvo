import json
import unittest

from strataevo.evolution.diagnosis import Diagnosis, DiagnosisReport, EvolutionLayer
from strataevo.evolution.plan import (
    EvolutionPlanner,
    MetricDirection,
    observe_expected_outcomes,
)
from strataevo.evolution.types import EvaluationReport
from tinyagent import Message, ModelResponse, ScriptedModel, Usage


class EvolutionPlanTests(unittest.TestCase):
    def test_planner_selects_one_diagnosis_with_measurable_outcomes(self):
        model = ScriptedModel(
            [ModelResponse(Message("assistant", json.dumps(self._plan())), Usage(50, 20))]
        )

        report = EvolutionPlanner(model).create_plan(
            self._diagnosis(),
            self._parent(),
            mutable_paths=["src/tinyagent"],
            existing_files=["src/tinyagent/agent.py", "src/tinyagent/workspace.py"],
        )

        self.assertEqual(report.plan.target_diagnosis, 1)
        self.assertEqual(report.plan.primary_layer, EvolutionLayer.ARCHITECTURE)
        self.assertEqual(report.plan.expected_outcomes[0].metric, "signal:artifact_missing")
        self.assertEqual(report.input_tokens, 50)
        request = model.requests[0][1].content
        self.assertIn('"available_metrics"', request)
        self.assertIn('"signal:artifact_missing": 2.0', request)
        self.assertIn('"src/tinyagent/workspace.py"', request)

    def test_likely_files_must_be_inside_mutable_paths(self):
        invalid = self._plan()
        invalid["likely_files"] = ["src/core/execution_engine.py"]
        model = ScriptedModel(
            [
                Message("assistant", json.dumps(invalid)),
                Message("assistant", json.dumps(self._plan())),
            ]
        )

        report = EvolutionPlanner(model).create_plan(
            self._diagnosis(), self._parent(), mutable_paths=["src/tinyagent"]
        )

        self.assertEqual(len(report.attempts), 2)
        self.assertIn("outside mutable paths", model.requests[1][-1].content)

    def test_plan_layer_must_match_selected_diagnosis(self):
        invalid = self._plan()
        invalid["primary_layer"] = "tools"
        model = ScriptedModel([Message("assistant", json.dumps(invalid))])

        with self.assertRaisesRegex(ValueError, "does not match"):
            EvolutionPlanner(model, repair_retries=0).create_plan(self._diagnosis(), self._parent())

    def test_unavailable_metric_is_repaired_once(self):
        invalid = self._plan()
        invalid["expected_outcomes"][0]["metric"] = "future_capability"
        model = ScriptedModel(
            [
                Message("assistant", json.dumps(invalid)),
                Message("assistant", json.dumps(self._plan())),
            ]
        )

        report = EvolutionPlanner(model).create_plan(self._diagnosis(), self._parent())

        self.assertEqual(len(report.attempts), 2)
        self.assertIn("unavailable metric", model.requests[1][-1].content)

    def test_missing_direction_is_repaired_once(self):
        invalid = self._plan()
        del invalid["expected_outcomes"][0]["direction"]
        model = ScriptedModel(
            [
                Message("assistant", json.dumps(invalid)),
                Message("assistant", json.dumps(self._plan())),
            ]
        )

        report = EvolutionPlanner(model).create_plan(self._diagnosis(), self._parent())

        self.assertEqual(len(report.attempts), 2)
        self.assertIn("invalid direction", model.requests[1][-1].content)

    def test_expected_outcomes_are_compared_with_candidate_metrics(self):
        model = ScriptedModel([Message("assistant", json.dumps(self._plan()))])
        plan = EvolutionPlanner(model).create_plan(self._diagnosis(), self._parent()).plan
        candidate = EvaluationReport(
            task_score=0.5,
            utility=0.495,
            metrics={"evidence_signal_counts": {}},
            output_dir="candidate",
            log_path="candidate.log",
        )

        observations = observe_expected_outcomes(
            plan, self._parent().to_dict(), candidate.to_dict()
        )

        self.assertEqual(observations[0]["direction"], MetricDirection.DECREASE.value)
        self.assertEqual(observations[0]["parent_value"], 2.0)
        self.assertEqual(observations[0]["candidate_value"], 0.0)
        self.assertTrue(observations[0]["satisfied"])

    @staticmethod
    def _plan() -> dict:
        return {
            "target_diagnosis": 1,
            "primary_layer": "architecture",
            "hypothesis": "Completion lacks a required-artifact check.",
            "intervention": "Validate required artifacts before accepting completion.",
            "expected_outcomes": [
                {
                    "metric": "signal:artifact_missing",
                    "direction": "decrease",
                    "reason": "The intervention directly targets missing artifacts.",
                },
                {
                    "metric": "task_score",
                    "direction": "non_decreasing",
                    "reason": "Task quality must be preserved.",
                },
            ],
            "likely_files": ["src/tinyagent/agent.py"],
            "expected_long_term_value": "Reusable completion validation for future tasks.",
            "prerequisites": ["Tasks expose their required artifacts."],
            "confidence": 0.85,
        }

    @staticmethod
    def _diagnosis() -> DiagnosisReport:
        return DiagnosisReport(
            source_dir="evaluation",
            input_case_count=2,
            diagnoses=[
                Diagnosis(
                    primary_layer=EvolutionLayer.CONTEXT,
                    related_layers=[],
                    problem="The prompt is ambiguous.",
                    evidence=["HumanEval/0 used unnecessary steps."],
                    affected_tasks=["HumanEval/0"],
                    proposed_direction="Clarify completion requirements.",
                    confidence=0.5,
                ),
                Diagnosis(
                    primary_layer=EvolutionLayer.ARCHITECTURE,
                    related_layers=[EvolutionLayer.TOOLS],
                    problem="The agent completes without its required artifact.",
                    evidence=["HumanEval/1 completed without solution.py."],
                    affected_tasks=["HumanEval/1"],
                    proposed_direction="Check required artifacts before completion.",
                    confidence=0.9,
                ),
            ],
            input_tokens=10,
            output_tokens=10,
            raw_output="{}",
            attempts=["{}"],
        )

    @staticmethod
    def _parent() -> EvaluationReport:
        return EvaluationReport(
            task_score=0.5,
            utility=0.49,
            metrics={
                "average_agent_steps": 5.0,
                "evidence_signal_counts": {"artifact_missing": 2},
            },
            output_dir="parent",
            log_path="parent.log",
        )


if __name__ == "__main__":
    unittest.main()
