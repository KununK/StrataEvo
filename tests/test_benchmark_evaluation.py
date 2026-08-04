import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.runtime.evaluation import BenchmarkEvaluator, create_evaluator
from strataevo.evolution.types import EvolutionConfig


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_registry_builds_humaneval_and_mbpp_evaluators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, display_name, module in (
                ("humaneval", "HumanEval", "eval.humaneval.run"),
                ("mbpp", "MBPP", "eval.mbpp.run"),
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


if __name__ == "__main__":
    unittest.main()
