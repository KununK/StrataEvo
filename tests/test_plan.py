import json
import unittest
from types import SimpleNamespace

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
        self.assertIn('"requires_strict_improvement": true', request)
        self.assertIn('"parent_task_score": 0.5', request)

    def test_planner_receives_explicit_refuted_hypotheses(self):
        model = ScriptedModel([Message("assistant", json.dumps(self._plan()))])
        history = [
            SimpleNamespace(
                generation=1,
                plan={
                    "primary_layer": "architecture",
                    "hypothesis": "old hypothesis",
                    "intervention": "old mechanism",
                },
                outcome=SimpleNamespace(
                    hypothesis_verdict="refuted",
                    counterevidence=["task score regressed"],
                    intervention_verdict="ineffective",
                    intervention_evidence=["candidate regressed"],
                ),
                to_context_dict=lambda: {"generation": 1},
            )
        ]

        EvolutionPlanner(model).create_plan(
            self._diagnosis(),
            self._parent(),
            history,
            mutable_paths=["src/tinyagent"],
        )

        request = model.requests[0][1].content
        self.assertIn('"refuted_hypotheses"', request)
        self.assertIn('"hypothesis": "old hypothesis"', request)
        self.assertIn('"counterevidence": [', request)
        self.assertIn('"task score regressed"', request)

    def test_planner_receives_failed_interventions_separately(self):
        model = ScriptedModel([Message("assistant", json.dumps(self._plan()))])
        history = [
            SimpleNamespace(
                generation=1,
                plan={
                    "primary_layer": "architecture",
                    "hypothesis": "artifact persistence is unreliable",
                    "intervention": "rewrite the agent loop",
                },
                outcome=SimpleNamespace(
                    hypothesis_verdict="untested",
                    counterevidence=[],
                    intervention_verdict="failed",
                    intervention_evidence=["candidate failed validation"],
                ),
                to_context_dict=lambda: {"generation": 1},
            )
        ]

        EvolutionPlanner(model).create_plan(
            self._diagnosis(),
            self._parent(),
            history,
            mutable_paths=["src/tinyagent"],
        )

        request = model.requests[0][1].content
        self.assertIn('"failed_interventions"', request)
        self.assertIn('"hypothesis": "artifact persistence is unreliable"', request)
        self.assertIn('"intervention": "rewrite the agent loop"', request)
        self.assertIn('"verdict": "failed"', request)
        self.assertIn('"candidate failed validation"', request)

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

    def test_forced_model_layer_selects_model_diagnosis(self):
        diagnosis = self._diagnosis()
        diagnosis.diagnoses.append(
            Diagnosis(
                primary_layer=EvolutionLayer.MODEL,
                related_layers=[],
                problem="The model produces incorrect algorithms.",
                evidence=["HumanEval/2 failed its assertions."],
                affected_tasks=["HumanEval/2"],
                proposed_direction="Adapt the model from verified trajectories.",
                confidence=0.8,
            )
        )
        plan = self._plan()
        plan.update(
            {
                "target_diagnosis": 2,
                "primary_layer": "model",
                "likely_files": [],
            }
        )
        model = ScriptedModel([Message("assistant", json.dumps(plan))])

        report = EvolutionPlanner(model).create_plan(
            diagnosis,
            self._parent(),
            model_evolution=True,
            force_layer=EvolutionLayer.MODEL,
        )

        self.assertEqual(report.plan.primary_layer, EvolutionLayer.MODEL)
        self.assertIn('"forced_layer": "model"', model.requests[0][1].content)

    def test_forced_layer_requires_matching_diagnosis(self):
        model = ScriptedModel([])
        with self.assertRaisesRegex(ValueError, "contains no 'model' problem"):
            EvolutionPlanner(model).create_plan(
                self._diagnosis(),
                self._parent(),
                model_evolution=True,
                force_layer=EvolutionLayer.MODEL,
            )

    def test_context_plan_does_not_require_source_files(self):
        diagnosis = DiagnosisReport(
            source_dir="parent",
            input_case_count=1,
            diagnoses=[
                Diagnosis(
                    primary_layer=EvolutionLayer.CONTEXT,
                    related_layers=[],
                    problem="The reusable instructions are unclear.",
                    evidence=["One task stopped without the required artifact."],
                    affected_tasks=["task/1"],
                    proposed_direction="Clarify the completion instruction.",
                    confidence=0.8,
                )
            ],
            input_tokens=0,
            output_tokens=0,
            raw_output="{}",
            attempts=["{}"],
        )
        plan = self._plan()
        plan.update(
            {
                "target_diagnosis": 0,
                "primary_layer": "context",
                "likely_files": [],
            }
        )
        model = ScriptedModel([Message("assistant", json.dumps(plan))])

        report = EvolutionPlanner(model).create_plan(
            diagnosis,
            self._parent(),
            force_layer=EvolutionLayer.CONTEXT,
        )

        self.assertEqual(report.plan.primary_layer, EvolutionLayer.CONTEXT)
        self.assertEqual(report.plan.likely_files, [])

    def test_tool_plan_does_not_require_source_files(self):
        diagnosis = DiagnosisReport(
            source_dir="parent",
            input_case_count=1,
            diagnoses=[
                Diagnosis(
                    primary_layer=EvolutionLayer.TOOLS,
                    related_layers=[],
                    problem="Tool descriptions do not explain intended sequencing.",
                    evidence=["The agent edited before reading."],
                    affected_tasks=["task/1"],
                    proposed_direction="Clarify tool sequencing.",
                    confidence=0.8,
                )
            ],
            input_tokens=0,
            output_tokens=0,
            raw_output="{}",
            attempts=["{}"],
        )
        plan = self._plan()
        plan.update(
            {
                "target_diagnosis": 0,
                "primary_layer": "tools",
                "likely_files": ["src/tinyagent/tool.py"],
            }
        )
        model = ScriptedModel([Message("assistant", json.dumps(plan))])

        report = EvolutionPlanner(model).create_plan(
            diagnosis, self._parent(), force_layer=EvolutionLayer.TOOLS
        )

        self.assertEqual(report.plan.primary_layer, EvolutionLayer.TOOLS)
        self.assertEqual(report.plan.likely_files, [])

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

    def test_plan_can_measure_an_intermediate_metric(self):
        plan = self._plan()
        plan["expected_outcomes"] = [plan["expected_outcomes"][0]]
        model = ScriptedModel([Message("assistant", json.dumps(plan))])

        report = EvolutionPlanner(model).create_plan(self._diagnosis(), self._parent())

        self.assertEqual(
            [outcome.metric for outcome in report.plan.expected_outcomes],
            ["signal:artifact_missing"],
        )

    def test_expected_outcomes_are_compared_with_candidate_metrics(self):
        model = ScriptedModel([Message("assistant", json.dumps(self._plan()))])
        plan = EvolutionPlanner(model).create_plan(self._diagnosis(), self._parent()).plan
        candidate = EvaluationReport(
            task_score=0.5,
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
                    "direction": "increase",
                    "reason": "The intervention must improve the strict promotion metric.",
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
            metrics={
                "average_agent_steps": 5.0,
                "evidence_signal_counts": {"artifact_missing": 2},
            },
            output_dir="parent",
            log_path="parent.log",
        )


if __name__ == "__main__":
    unittest.main()
