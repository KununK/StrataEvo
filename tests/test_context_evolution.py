import json
import tempfile
import unittest
from pathlib import Path

from eval.coding_agent import SYSTEM_PROMPT, USER_PROMPT, load_context_prompts
from strataevo.evolution.decision import EvolutionDecision, EvolutionLayer
from strataevo.evolution.layers.context import (
    ContextEvolver,
    ContextProfile,
    evolve_context,
)
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Message, ModelResponse, ScriptedModel, Usage


class FakeEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.contexts = []
        self.evaluations = []

    def set_context(self, context):
        self.contexts.append(context)

    def evaluate(self, output_dir):
        self.evaluations.append(str(output_dir))
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
            ContextProfile(), self._decision(), []
        )

        self.assertIn("required artifact", profile.system_prompt_addendum)
        self.assertEqual(metadata["input_tokens"], 20)
        self.assertEqual(metadata["output_tokens"], 10)
        request = model.requests[0][1].content
        self.assertIn("parent_context", request)

    def test_empty_context_candidate_is_rejected(self):
        model = ScriptedModel([Message("assistant", json.dumps({"system_prompt_addendum": ""}))])

        with self.assertRaisesRegex(ValueError, "at least one addendum"):
            ContextEvolver(model, repair_retries=0).create_candidate(
                ContextProfile(), self._decision(), []
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
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=1),
                1,
                generation_dir,
                parent_report,
                evaluator,
                self._decision(),
                [],
                parent_context=None,
                model_name="test-model",
                model=model,
            )

            self.assertEqual(result.decision, "accepted")
            self.assertEqual(result.candidate_report.task_score, 0.6)
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
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=1),
                1,
                Path(directory) / "generation-0001",
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._decision(),
                [],
                parent_context=parent,
                model_name="test-model",
                model=model,
            )

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(evaluator.contexts[-1], parent)

    def test_rejected_candidate_is_refined_with_screening_feedback(self):
        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    json.dumps(
                        {
                            "system_prompt_addendum": "Use broad verification.",
                            "task_prompt_addendum": "",
                        }
                    ),
                ),
                Message(
                    "assistant",
                    json.dumps(
                        {
                            "system_prompt_addendum": "Verify only the requested behavior.",
                            "task_prompt_addendum": "",
                        }
                    ),
                ),
            ]
        )
        evaluator = FakeEvaluator([0.4, 0.6, 0.5, 0.7])
        with tempfile.TemporaryDirectory() as directory:
            generation_dir = Path(directory) / "generation-0001"
            result = evolve_context(
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=2),
                1,
                generation_dir,
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._decision(),
                [],
                parent_context=None,
                model_name="test-model",
                model=model,
            )

            feedback = json.loads(
                (generation_dir / "context/refinement.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.decision, "accepted")
        self.assertEqual(len(feedback), 2)
        self.assertIn('"score_delta"', model.requests[1][1].content)
        self.assertIn("Use broad verification.", model.requests[1][1].content)

    def test_refinement_restores_parent_and_reports_best_rejected_candidate(self):
        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    json.dumps({"system_prompt_addendum": "First attempt."}),
                ),
                Message(
                    "assistant",
                    json.dumps({"system_prompt_addendum": "Second attempt."}),
                ),
            ]
        )
        evaluator = FakeEvaluator([0.4, 0.3])
        parent = {
            "system_prompt_addendum": "Parent.",
            "task_prompt_addendum": "",
            "path": "parent.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = evolve_context(
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=2),
                1,
                Path(directory) / "generation-0001",
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._decision(),
                [],
                parent_context=parent,
                model_name="test-model",
                model=model,
            )

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(result.candidate_report.task_score, 0.4)
        self.assertEqual(result.candidate["system_prompt_addendum"], "First attempt.")
        self.assertEqual(evaluator.contexts[-1], parent)

    def test_duplicate_candidates_are_not_re_evaluated(self):
        response = json.dumps({"system_prompt_addendum": "Same candidate."})
        model = ScriptedModel([Message("assistant", response) for _ in range(3)])
        evaluator = FakeEvaluator([0.4])
        with tempfile.TemporaryDirectory() as directory:
            generation_dir = Path(directory) / "generation-0001"
            result = evolve_context(
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=3),
                1,
                generation_dir,
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._decision(),
                [],
                parent_context=None,
                model_name="test-model",
                model=model,
            )
            feedback = json.loads(
                (generation_dir / "context/refinement.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(len(evaluator.evaluations), 1)
        self.assertEqual(
            [item["outcome_type"] for item in feedback],
            ["evaluated", "duplicate_candidate", "duplicate_candidate"],
        )

    def test_single_screening_uses_recorded_parent(self):
        model = ScriptedModel(
            [Message("assistant", json.dumps({"system_prompt_addendum": "Candidate."}))]
        )
        evaluator = FakeEvaluator([0.6])
        parent = {
            "system_prompt_addendum": "Parent.",
            "task_prompt_addendum": "",
            "path": "parent.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = evolve_context(
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=1),
                1,
                Path(directory) / "generation-0001",
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._decision(),
                [],
                parent_context=parent,
                model_name="test-model",
                model=model,
            )

        self.assertEqual(result.decision, "accepted")
        self.assertEqual(result.candidate_report.task_score, 0.6)
        self.assertEqual(len(evaluator.evaluations), 1)
        self.assertEqual(evaluator.contexts[-1], result.candidate)

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
    def _decision():
        return EvolutionDecision(
            EvolutionLayer.CONTEXT,
            ["A candidate was missing."],
            ["task/1"],
            "A clearer reusable instruction will reduce missing artifacts.",
            "Add one concise prompt instruction.",
            [],
        )


if __name__ == "__main__":
    unittest.main()
