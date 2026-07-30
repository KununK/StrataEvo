import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strataevo.evolution.diagnosis import Diagnosis, EvolutionLayer
from strataevo.evolution.model_evolution import (
    build_training_dataset,
    evolve_model,
    task_changes,
)
from strataevo.evolution.plan import EvolutionPlan, ExpectedOutcome, MetricDirection
from strataevo.evolution.repair import (
    RepairCollection,
    _benchmark_adapter,
    collect_failed_task_repairs,
)
from strataevo.evolution.sft import _chat_ids
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Message, ScriptedModel, ToolCall


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
    def test_cli_import_does_not_require_repository_on_python_path(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from strataevo.evolution.cli import main; assert callable(main)",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_chat_ids_accepts_transformers_batch_encoding_shape(self):
        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        self.assertEqual(
            _chat_ids(FakeTokenizer(), [{"role": "user", "content": "test"}]),
            [1, 2, 3],
        )

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
            diagnosis, plan = self._model_direction()

            repairs = RepairCollection(
                failed_tasks=["failed"],
                repaired_tasks=["failed"],
                still_failed_tasks=[],
                attempts=1,
                successful_trajectories=1,
                output_dir=str(root / "repairs"),
            )
            with (
                patch(
                    "strataevo.evolution.model_evolution.collect_failed_task_repairs",
                    return_value=repairs,
                ),
                patch(
                    "strataevo.evolution.model_evolution.build_training_dataset",
                    return_value={
                        "examples": 2,
                        "repair_examples": 1,
                        "replay_examples": 1,
                        "trained_repair_tasks": ["failed"],
                    },
                ),
                patch("strataevo.evolution.model_evolution._run_training"),
            ):
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
                    diagnosis=diagnosis,
                    plan=plan,
                )

            self.assertEqual(result.decision, "accepted")
            self.assertEqual(result.candidate_report.task_score, 0.7)
            self.assertEqual(result.candidate["training_examples"], 2)
            self.assertEqual(
                result.candidate["repair_collection"]["repaired_tasks"],
                ["failed"],
            )
            self.assertEqual(evaluator.models[-1], result.candidate["name"])
            self.assertEqual(runtime.events[-1][0], "activate")
            self.assertEqual(runtime.events[0], ("deactivate", "parent-adapter"))
            self.assertEqual(
                result.candidate["planned_intervention"]["affected_tasks"],
                ["failed"],
            )
            self.assertEqual(
                result.candidate["executed_intervention"]["targeted_tasks"],
                ["failed"],
            )
            self.assertEqual(
                result.candidate["executed_intervention"]["training"]["epochs"],
                1,
            )
            train_config = json.loads(
                (
                    root / "generation-0001/model/train_config.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(train_config["epochs"], 1)
            self.assertNotIn("max_steps", train_config)

    def test_training_dataset_prioritizes_repairs_then_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            repairs_dir = root / "repairs"
            parent.mkdir()
            repairs_dir.mkdir()
            self._write_jsonl(
                parent / "results.jsonl",
                [{"task_id": "passed", "passed": True}],
            )
            self._write_jsonl(
                parent / "generations.jsonl",
                [{"task_id": "passed", "messages": [{"role": "assistant", "content": "old"}]}],
            )
            self._write_jsonl(
                repairs_dir / "results.jsonl",
                [
                    {"task_id": "failed", "repair_attempt": 1, "passed": True},
                    {"task_id": "failed", "repair_attempt": 2, "passed": True},
                ],
            )
            self._write_jsonl(
                repairs_dir / "generations.jsonl",
                [
                    {
                        "task_id": "failed",
                        "repair_attempt": 1,
                        "messages": [{"role": "assistant", "content": "fixed"}],
                    },
                    {
                        "task_id": "failed",
                        "repair_attempt": 2,
                        "messages": [{"role": "assistant", "content": "duplicate"}],
                    },
                ],
            )
            collection = RepairCollection(
                ["failed"], ["failed"], [], 2, 2, str(repairs_dir)
            )
            destination = root / "training.jsonl"

            summary = build_training_dataset(parent, collection, destination, limit=2)

            rows = [
                json.loads(line)
                for line in destination.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["source"] for row in rows], ["repair", "replay"])
            self.assertEqual(summary["trained_repair_tasks"], ["failed"])

    def test_repair_collector_runs_failed_task_through_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            self._write_jsonl(
                evaluation / "results.jsonl",
                [
                    {"task_id": "task", "passed": False, "status": "assertion_error"},
                    {"task_id": "other", "passed": False, "status": "assertion_error"},
                ],
            )
            model = ScriptedModel(
                [
                    Message(
                        "assistant",
                        tool_calls=[
                            ToolCall(
                                "write",
                                "write_file",
                                {"path": "solution.py", "content": "def answer(): return 1\n"},
                            )
                        ],
                    ),
                    Message("assistant", "repaired"),
                ]
            )
            config = EvolutionConfig(
                repo=str(Path(__file__).resolve().parents[1]),
                run_name="repair",
                benchmark="mbpp",
                repair_attempts=1,
                eval_workers=1,
            )
            adapter = (
                [
                    {"task_id": "task", "prompt": "repair", "entry_point": "answer"},
                    {"task_id": "other", "prompt": "repair", "entry_point": "answer"},
                ],
                lambda _task: "def answer(): ...\n",
                lambda _task, source, _timeout: {
                    "passed": "return 1" in source,
                    "status": "pass",
                },
            )

            with patch(
                "strataevo.evolution.repair._benchmark_adapter",
                return_value=adapter,
            ):
                collection = collect_failed_task_repairs(
                    config,
                    evaluation,
                    root / "repairs",
                    model_name="test",
                    model=model,
                    task_ids={"task"},
                    guidance="Focus on preserving the function contract.",
                )

            self.assertEqual(collection.failed_tasks, ["task"])
            self.assertEqual(collection.repaired_tasks, ["task"])
            self.assertEqual(collection.still_failed_tasks, [])
            self.assertEqual(collection.successful_trajectories, 1)
            self.assertIn(
                "Focus on preserving the function contract.",
                model.requests[0][1].content,
            )

    def test_repair_adapter_reuses_saved_task_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"dataset": "unused"}),
                encoding="utf-8",
            )
            task = {
                "task_id": "11",
                "prompt": "add",
                "entry_point": "add",
                "test_list": ["assert add(1, 2) == 3"],
                "test_setup": "",
                "test_imports": "",
            }
            self._write_jsonl(root / "tasks.jsonl", [task])

            repo = Path(__file__).resolve().parents[1]
            tasks, render, _verify = _benchmark_adapter(
                "mbpp",
                root / "config.json",
                repo,
            )

            self.assertEqual(tasks, [task])
            self.assertIn("TESTS =", render(tasks[0]))

    def test_task_changes_records_fixed_regressed_and_still_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            candidate = root / "candidate"
            parent.mkdir()
            candidate.mkdir()
            self._write_jsonl(
                parent / "results.jsonl",
                [
                    {"task_id": "fixed", "passed": False},
                    {"task_id": "regressed", "passed": True},
                    {"task_id": "failed", "passed": False},
                ],
            )
            self._write_jsonl(
                candidate / "results.jsonl",
                [
                    {"task_id": "fixed", "passed": True},
                    {"task_id": "regressed", "passed": False},
                    {"task_id": "failed", "passed": False},
                ],
            )

            changes = task_changes(parent, candidate)

            self.assertEqual(changes["fixed_tasks"], ["fixed"])
            self.assertEqual(changes["regressed_tasks"], ["regressed"])
            self.assertEqual(changes["still_failed_tasks"], ["failed"])

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _model_direction():
        diagnosis = Diagnosis(
            primary_layer=EvolutionLayer.MODEL,
            related_layers=[],
            problem="The model violates the requested function contract.",
            evidence=["failed did not satisfy the verifier"],
            affected_tasks=["failed"],
            proposed_direction="Improve contract-aware reasoning.",
            confidence=0.8,
        )
        plan = EvolutionPlan(
            target_diagnosis=0,
            primary_layer=EvolutionLayer.MODEL,
            hypothesis="Contract-aware repair trajectories improve task success.",
            intervention="Train on verified repairs for the affected failed tasks.",
            expected_outcomes=[
                ExpectedOutcome(
                    "task_score",
                    MetricDirection.INCREASE,
                    "The selected failures should be repaired.",
                )
            ],
            likely_files=[],
            expected_long_term_value="Improve related coding tasks.",
            prerequisites=[],
            confidence=0.8,
        )
        return diagnosis, plan


if __name__ == "__main__":
    unittest.main()
