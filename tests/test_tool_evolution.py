import json
import tempfile
import unittest
from pathlib import Path

from eval.coding_agent import load_tool_profile
from strataevo.evolution.decision import EvolutionDecision, EvolutionLayer
from strataevo.evolution.layers.tools import ToolEvolver, ToolProfile, evolve_tools
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Message, ModelResponse, ScriptedModel, Usage


class FakeEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.profiles = []

    def set_tool_profile(self, profile):
        self.profiles.append(profile)

    def available_tool_descriptions(self):
        return {"read_file": "Read a UTF-8 text file."}

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
            self._decision(),
            [],
        )
        self.assertIn("read_file", profile.description_addenda)
        self.assertEqual(metadata["input_tokens"], 20)
        self.assertIn("available_tools", model.requests[0][1].content)

    def test_evolver_explains_dynamic_tool_wildcard(self):
        model = ScriptedModel(
            [Message("assistant", json.dumps({"description_addenda": {"*": "Check state."}}))]
        )

        profile, _metadata = ToolEvolver(model).create_candidate(
            ToolProfile({}),
            {"*": "Task-specific tools."},
            self._decision(),
            [],
        )

        self.assertEqual(profile.description_addenda, {"*": "Check state."})
        self.assertIn('only key is "*"', model.requests[0][0].content)

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
                self._decision(),
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
                self._decision(),
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
                    json.dumps({"description_addenda": {"read_file": "Read every file first."}}),
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
                self._decision(),
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
    def _decision():
        return EvolutionDecision(
            EvolutionLayer.TOOLS,
            ["The first call was write_file."],
            ["task/1"],
            "Clearer guidance will improve evidence use.",
            "Append concise sequencing guidance.",
            [],
        )


if __name__ == "__main__":
    unittest.main()
