import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strataevo.evolution.model_evolution import (
    collect_verified_trajectories,
    evolve_model,
)
from strataevo.evolution.sft import _chat_ids
from strataevo.evolution.types import EvaluationReport, EvolutionConfig


class FakeRuntime:
    def __init__(self):
        self.events = []

    def activate(self, name, path):
        self.events.append(("activate", name, str(path)))

    def deactivate(self, name):
        self.events.append(("deactivate", name))


class FakeEvaluator:
    def __init__(self, reports):
        self.reports = iter(reports)
        self.models = []

    def set_model(self, model):
        self.models.append(model)

    def evaluate(self, output_dir):
        report = next(self.reports)
        return EvaluationReport(
            report.task_score,
            report.metrics,
            str(output_dir),
            report.log_path,
        )


class ModelEvolutionTests(unittest.TestCase):
    def test_chat_ids_accepts_transformers_batch_encoding_shape(self):
        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        self.assertEqual(
            _chat_ids(FakeTokenizer(), [{"role": "user", "content": "test"}]),
            [1, 2, 3],
        )

    def test_collects_only_verifier_passing_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_jsonl(
                root / "results.jsonl",
                [
                    {"task_id": "pass", "passed": True},
                    {"task_id": "fail", "passed": False},
                ],
            )
            self._write_jsonl(
                root / "generations.jsonl",
                [
                    {"task_id": "pass", "messages": [{"role": "user", "content": "a"}]},
                    {"task_id": "fail", "messages": [{"role": "user", "content": "b"}]},
                ],
            )
            destination = root / "training.jsonl"

            count = collect_verified_trajectories(root, destination, limit=10)

            rows = [
                json.loads(line)
                for line in destination.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(count, 1)
            self.assertEqual(rows[0]["task_id"], "pass")

    def test_accepts_confirmed_lora_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "parent"
            evaluation.mkdir()
            self._write_jsonl(
                evaluation / "results.jsonl",
                [{"task_id": "task", "passed": True}],
            )
            self._write_jsonl(
                evaluation / "generations.jsonl",
                [
                    {
                        "task_id": "task",
                        "messages": [
                            {"role": "user", "content": "solve"},
                            {"role": "assistant", "content": "done"},
                        ],
                    }
                ],
            )
            config = EvolutionConfig(
                repo=str(root),
                run_name="model-test",
                model_evolution=True,
            )
            parent = EvaluationReport(0.5, {}, str(evaluation), "parent.log")
            reports = [
                EvaluationReport(0.6, {}, "", "screening.log"),
                EvaluationReport(0.5, {}, "", "fresh-parent.log"),
                EvaluationReport(0.7, {}, "", "confirmed.log"),
            ]
            evaluator = FakeEvaluator(reports)
            runtime = FakeRuntime()

            with patch("strataevo.evolution.model_evolution._run_training"):
                result = evolve_model(
                    config,
                    1,
                    root / "generation-0001",
                    parent,
                    evaluator,
                    parent_model="parent-adapter",
                    parent_adapter={
                        "name": "parent-adapter",
                        "path": str(root / "parent-adapter"),
                    },
                    runtime=runtime,
                )

            self.assertEqual(result.decision, "accepted")
            self.assertEqual(result.candidate_report.task_score, 0.7)
            self.assertEqual(result.candidate["training_examples"], 1)
            self.assertEqual(evaluator.models[-1], result.candidate["name"])
            self.assertEqual(runtime.events[-1][0], "activate")
            self.assertEqual(runtime.events[0], ("deactivate", "parent-adapter"))

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
