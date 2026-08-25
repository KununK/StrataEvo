import tempfile
import unittest
from pathlib import Path

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

    def test_bfcl_rejects_unsupported_model_evolution(self):
        args = parse_args(["--benchmark", "bfcl", "--enable-model-evolution"])

        with self.assertRaisesRegex(ValueError, "bfcl does not support model evolution"):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
