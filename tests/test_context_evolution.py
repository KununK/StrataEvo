import json
import tempfile
import unittest
from pathlib import Path

from eval.coding_agent import SYSTEM_PROMPT, USER_PROMPT, load_context_prompts
from strataevo.evolution.context_evolution import (
    ContextEvolver,
    ContextProfile,
    evolve_context,
)
from strataevo.evolution.contract import EvaluationContract
from strataevo.evolution.diagnosis import Diagnosis, DiagnosisReport, EvolutionLayer
from strataevo.evolution.plan import (
    EvolutionPlan,
    EvolutionPlanReport,
    ExpectedOutcome,
    MetricDirection,
)
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Message, ModelResponse, ScriptedModel, Usage

CONTRACT = EvaluationContract(
    benchmark="test",
    objective="improve task score",
    direct_paths=("src/tinyagent",),
    deferred_paths=(),
)


class FakeEvaluator:
    contract = CONTRACT

    def __init__(self, scores):
        self.scores = iter(scores)
        self.contexts = []

    def set_context(self, context):
        self.contexts.append(context)

    def evaluate(self, output_dir):
        score = next(self.scores)
        return EvaluationReport(score, {}, str(output_dir), "evaluation.log")


class ContextEvolutionTests(unittest.TestCase):
    def test_context_evolver_returns_a_versionable_general_profile(self):
        model = ScriptedModel(
            [
                ModelResponse(
                    Message(
                        "assistant",
                        json.dumps(
                            {
                                "system_prompt_addendum": (
                                    "Inspect the required artifact before finishing."
                                ),
                                "task_prompt_addendum": "Use one focused verification step.",
                            }
                        ),
                    ),
                    Usage(20, 10),
                )
            ]
        )

        profile, metadata = ContextEvolver(model).create_candidate(
            ContextProfile(), self._diagnosis(), self._plan(), [], CONTRACT
        )

        self.assertIn("required artifact", profile.system_prompt_addendum)
        self.assertEqual(metadata["input_tokens"], 20)
        self.assertEqual(metadata["output_tokens"], 10)
        request = model.requests[0][1].content
        self.assertIn("parent_context", request)
        self.assertIn("evaluation_contract", request)

    def test_empty_context_candidate_is_rejected(self):
        model = ScriptedModel([Message("assistant", json.dumps({"system_prompt_addendum": ""}))])

        with self.assertRaisesRegex(ValueError, "at least one addendum"):
            ContextEvolver(model, repair_retries=0).create_candidate(
                ContextProfile(), self._diagnosis(), self._plan(), [], CONTRACT
            )

    def test_accepted_candidate_remains_active_and_is_persisted(self):
        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    json.dumps(
                        {
                            "system_prompt_addendum": "Check the requested callable name.",
                            "task_prompt_addendum": "",
                        }
                    ),
                )
            ]
        )
        evaluator = FakeEvaluator([0.6, 0.5, 0.7])
        parent_report = EvaluationReport(0.5, {}, "parent", "parent.log")
        with tempfile.TemporaryDirectory() as directory:
            generation_dir = Path(directory) / "generation-0001"
            result = evolve_context(
                EvolutionConfig(repo=directory, run_name="test"),
                1,
                generation_dir,
                parent_report,
                evaluator,
                self._diagnosis(),
                self._plan(),
                [],
                parent_context=None,
                model_name="test-model",
                model=model,
            )

            self.assertEqual(result.decision, "accepted")
            self.assertEqual(result.candidate_report.task_score, 0.7)
            self.assertTrue(Path(result.candidate["path"]).is_file())
            self.assertEqual(evaluator.contexts[-1], result.candidate)
            candidate = json.loads(
                (generation_dir / "context/candidate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(candidate["generation"], 1)

    def test_rejected_candidate_restores_parent_context(self):
        parent = {
            "system_prompt_addendum": "Keep the existing behavior.",
            "task_prompt_addendum": "",
            "path": "parent.json",
        }
        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    json.dumps(
                        {
                            "system_prompt_addendum": "Try an alternative behavior.",
                            "task_prompt_addendum": "",
                        }
                    ),
                )
            ]
        )
        evaluator = FakeEvaluator([0.4])
        with tempfile.TemporaryDirectory() as directory:
            result = evolve_context(
                EvolutionConfig(repo=directory, run_name="test"),
                1,
                Path(directory) / "generation-0001",
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._diagnosis(),
                self._plan(),
                [],
                parent_context=parent,
                model_name="test-model",
                model=model,
            )

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(evaluator.contexts[-1], parent)

    def test_benchmark_prompt_loader_appends_both_context_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(
                json.dumps(
                    {
                        "system_prompt_addendum": "System addition.",
                        "task_prompt_addendum": "Task addition.",
                    }
                ),
                encoding="utf-8",
            )
            system, user = load_context_prompts(str(path))

        self.assertEqual(system, SYSTEM_PROMPT + "\n\nSystem addition.")
        self.assertEqual(user, USER_PROMPT + "\n\nTask addition.")

    @staticmethod
    def _diagnosis():
        return DiagnosisReport(
            source_dir="parent",
            input_case_count=1,
            diagnoses=[
                Diagnosis(
                    primary_layer=EvolutionLayer.CONTEXT,
                    related_layers=[],
                    problem="Instructions do not focus the agent on the requested artifact.",
                    evidence=["A candidate was missing."],
                    affected_tasks=["task/1"],
                    proposed_direction="Clarify the reusable completion policy.",
                    confidence=0.8,
                )
            ],
            input_tokens=0,
            output_tokens=0,
            raw_output="{}",
            attempts=["{}"],
        )

    @staticmethod
    def _plan():
        return EvolutionPlanReport(
            plan=EvolutionPlan(
                target_diagnosis=0,
                primary_layer=EvolutionLayer.CONTEXT,
                hypothesis="A clearer reusable instruction will reduce missing artifacts.",
                intervention="Add one concise prompt instruction.",
                expected_outcomes=[
                    ExpectedOutcome(
                        metric="task_score",
                        direction=MetricDirection.INCREASE,
                        reason="The intervention should solve more tasks.",
                    )
                ],
                likely_files=[],
                expected_long_term_value="Reusable across coding tasks.",
                prerequisites=[],
                confidence=0.8,
            ),
            available_metrics={"task_score": 0.5},
            input_tokens=0,
            output_tokens=0,
            raw_output="{}",
            attempts=["{}"],
        )


if __name__ == "__main__":
    unittest.main()
