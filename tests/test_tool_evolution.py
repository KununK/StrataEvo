import json
import tempfile
import unittest
from pathlib import Path

from eval.coding_agent import load_tool_profile
from strataevo.evolution.diagnosis import Diagnosis, DiagnosisReport, EvolutionLayer
from strataevo.evolution.layers.tools import ToolEvolver, ToolProfile, evolve_tools
from strataevo.evolution.plan import (
    EvolutionPlan,
    EvolutionPlanReport,
    ExpectedOutcome,
    MetricDirection,
)
from strataevo.evolution.runtime.contract import EvaluationContract
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Message, ModelResponse, ScriptedModel, Usage

CONTRACT = EvaluationContract("test", "improve score", ("src/tinyagent",), ())


class FakeEvaluator:
    contract = CONTRACT

    def __init__(self, scores):
        self.scores = iter(scores)
        self.profiles = []

    def set_tool_profile(self, profile):
        self.profiles.append(profile)

    def evaluate(self, output_dir):
        return EvaluationReport(next(self.scores), {}, str(output_dir), "evaluation.log")


class ToolEvolutionTests(unittest.TestCase):
    def test_evolver_returns_a_known_tool_profile(self):
        model = ScriptedModel(
            [
                ModelResponse(
                    Message(
                        "assistant",
                        json.dumps(
                            {
                                "description_addenda": {
                                    "read_file": "Inspect the task before editing."
                                }
                            }
                        ),
                    ),
                    Usage(20, 10),
                )
            ]
        )
        profile, metadata = ToolEvolver(model).create_candidate(
            ToolProfile({}),
            {"read_file": "Read a file."},
            self._diagnosis(),
            self._plan(),
            [],
            CONTRACT,
        )
        self.assertIn("read_file", profile.description_addenda)
        self.assertEqual(metadata["input_tokens"], 20)
        self.assertIn("available_tools", model.requests[0][1].content)

    def test_unknown_tool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown tools"):
            ToolProfile.from_dict({"description_addenda": {"unknown": "guidance"}}, {"read_file"})

    def test_accepted_candidate_remains_active(self):
        evaluator = FakeEvaluator([0.6, 0.5, 0.7])
        with tempfile.TemporaryDirectory() as directory:
            result = evolve_tools(
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=1),
                1,
                Path(directory) / "generation-0001",
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._diagnosis(),
                self._plan(),
                [],
                parent_profile=None,
                model_name="test-model",
                model=self._model(),
            )
        self.assertEqual(result.decision, "accepted")
        self.assertEqual(evaluator.profiles[-1], result.candidate)

    def test_rejected_candidate_restores_parent(self):
        parent = {
            "description_addenda": {"read_file": "Keep current guidance."},
            "path": "parent.json",
        }
        evaluator = FakeEvaluator([0.4])
        with tempfile.TemporaryDirectory() as directory:
            result = evolve_tools(
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=1),
                1,
                Path(directory) / "generation-0001",
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._diagnosis(),
                self._plan(),
                [],
                parent_profile=parent,
                model_name="test-model",
                model=self._model(),
            )
        self.assertEqual(result.decision, "rejected")
        self.assertEqual(evaluator.profiles[-1], parent)

    def test_rejected_candidate_is_refined_with_screening_feedback(self):
        model = ScriptedModel(
            [
                Message(
                    "assistant",
                    json.dumps(
                        {"description_addenda": {"read_file": "Read every file first."}}
                    ),
                ),
                Message(
                    "assistant",
                    json.dumps(
                        {"description_addenda": {"read_file": "Read relevant evidence first."}}
                    ),
                ),
            ]
        )
        evaluator = FakeEvaluator([0.4, 0.6, 0.5, 0.7])
        with tempfile.TemporaryDirectory() as directory:
            generation_dir = Path(directory) / "generation-0001"
            result = evolve_tools(
                EvolutionConfig(repo=directory, run_name="test", max_eval_attempts=2),
                1,
                generation_dir,
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                evaluator,
                self._diagnosis(),
                self._plan(),
                [],
                parent_profile=None,
                model_name="test-model",
                model=model,
            )

        self.assertEqual(result.decision, "accepted")
        self.assertIn("Read every file first.", model.requests[1][1].content)

    def test_benchmark_loader_reads_description_addenda(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            path.write_text(
                json.dumps({"description_addenda": {"read_file": "Read first."}}), encoding="utf-8"
            )
            profile = load_tool_profile(str(path))
        self.assertEqual(profile, {"read_file": "Read first."})

    @staticmethod
    def _model():
        return ScriptedModel(
            [
                Message(
                    "assistant",
                    json.dumps(
                        {"description_addenda": {"read_file": "Inspect evidence before editing."}}
                    ),
                )
            ]
        )

    @staticmethod
    def _diagnosis():
        return DiagnosisReport(
            source_dir="parent",
            input_case_count=1,
            diagnoses=[
                Diagnosis(
                    primary_layer=EvolutionLayer.TOOLS,
                    related_layers=[],
                    problem="The agent edits before reading evidence.",
                    evidence=["The first call was write_file."],
                    affected_tasks=["task/1"],
                    proposed_direction="Clarify read-before-write sequencing.",
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
                primary_layer=EvolutionLayer.TOOLS,
                hypothesis="Clearer guidance will improve evidence use.",
                intervention="Append concise sequencing guidance.",
                expected_outcomes=[
                    ExpectedOutcome(
                        "task_score",
                        MetricDirection.INCREASE,
                        "Better evidence use should solve more tasks.",
                    )
                ],
                likely_files=[],
                expected_long_term_value="Reusable tool guidance.",
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
