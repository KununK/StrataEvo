import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def test_transfer_help_and_invalid_arguments_do_not_start_experiment(self):
        for argument, expected in (("--help", 0), ("--unknown", 2)):
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/train_transfer.sh"), argument],
                cwd="/tmp", capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, expected, result.stderr)

    def test_evaluation_help_and_invalid_benchmark_do_not_call_model(self):
        for argument, expected in (("--help", 0), ("unknown", 2)):
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/evaluate.sh"), argument],
                cwd="/tmp", capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, expected, result.stderr)

    def test_legacy_wrappers_forward_options_from_another_directory(self):
        for benchmark in ("humaneval", "mbpp", "bfcl"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                env = {
                    **os.environ, "PYTHON_BIN": "/bin/echo", "LIMIT": "5", "OFFSET": "7",
                    "WORKERS": "2", "RUN_NAME": "script-test", "FORCE_RERUN": "1",
                    "LOG_PATH": str(Path(directory) / "run.log"),
                    "OUTPUT_DIR": str(Path(directory) / "output"),
                }
                result = subprocess.run(
                    ["bash", str(ROOT / f"eval/run_{benchmark}.sh"), "--max-steps", "3"],
                    cwd=directory, env=env, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for expected in (f"-m eval.{benchmark}.run", "--limit 5", "--offset 7",
                                 "--workers 2", "--no-resume", "--max-steps 3"):
                    self.assertIn(expected, result.stdout)
                self.assertEqual((Path(directory) / "run.log").read_text(), result.stdout)

    def test_evaluation_failure_propagates_through_tee(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "PYTHON_BIN": "/bin/false",
                   "LOG_PATH": str(Path(directory) / "run.log")}
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/evaluate.sh"), "mbpp"],
                env=env, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
