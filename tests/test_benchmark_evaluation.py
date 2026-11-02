import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strataevo.evolution.cli import _validate_args, parse_args
from strataevo.evolution.runtime.evaluation import BenchmarkEvaluator, create_evaluator
from strataevo.evolution.types import EvolutionConfig


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_registry_builds_benchmark_evaluators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, display_name, module in (
                ("humaneval", "HumanEval", "eval.humaneval.run"),
                ("mbpp", "MBPP", "eval.mbpp.run"),
                ("bfcl", "BFCL V4 multi-turn", "eval.bfcl.run"),
            ):
                config = EvolutionConfig(repo=str(root), run_name="test", benchmark=name)
                evaluator = create_evaluator(root, config)

                self.assertIsInstance(evaluator, BenchmarkEvaluator)
                self.assertEqual(evaluator.spec.display_name, display_name)
                self.assertEqual(evaluator.spec.module, module)

    def test_registry_rejects_unknown_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = EvolutionConfig(repo=str(root), run_name="test", benchmark="unknown")

            with self.assertRaisesRegex(ValueError, "unknown benchmark"):
                create_evaluator(root, config)

    def test_bfcl_supports_model_evolution(self):
        args = parse_args(["--benchmark", "bfcl", "--enable-model-evolution"])
        _validate_args(args)

    def test_split_defaults_to_test_for_existing_configs(self):
        config = EvolutionConfig.from_dict({"repo": ".", "run_name": "old"})
        self.assertEqual(config.eval_split, "test")
        self.assertEqual(parse_args([]).eval_split, "test")

    def test_non_test_split_requires_mbpp(self):
        for benchmark in ("humaneval", "bfcl"):
            with self.subTest(benchmark=benchmark):
                args = parse_args(["--benchmark", benchmark, "--eval-split", "train"])
                with self.assertRaisesRegex(ValueError, "only for MBPP"):
                    _validate_args(args)
        _validate_args(parse_args(["--benchmark", "mbpp", "--eval-split", "train"]))

    def test_mbpp_evaluator_passes_selected_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = EvolutionConfig(repo=str(root), run_name="train", benchmark="mbpp",
                                     eval_split="train")
            evaluator = create_evaluator(root, config)
            with patch("strataevo.evolution.runtime.evaluation.run_commands",
                       return_value=(False, "captured")) as commands:
                with self.assertRaises(RuntimeError):
                    evaluator.evaluate(root / "evaluation")
            command = commands.call_args.args[0][0]
            self.assertEqual(command[command.index("--split") + 1], "train")


if __name__ == "__main__":
    unittest.main()
