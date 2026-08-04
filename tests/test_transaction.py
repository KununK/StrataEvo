import subprocess
import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.layers.architecture import CandidateEvaluationSession
from strataevo.evolution.runtime.repository import GitRepository
from strataevo.evolution.types import EvaluationReport


class FakeEvaluator:
    def evaluate(self, output_dir: Path) -> EvaluationReport:
        output_dir.mkdir(parents=True)
        return EvaluationReport(0.6, {"passed": 6}, str(output_dir), "evaluation.log")


class TransactionTests(unittest.TestCase):
    def test_executable_candidate_can_be_restored_and_rolled_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "tinyagent" / "agent.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            self._git(root, "init")
            self._git(root, "config", "user.email", "test@example.com")
            self._git(root, "config", "user.name", "Test")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "parent")

            repository = GitRepository(root, ["src/tinyagent"])
            source.write_text("VALUE = 2\n", encoding="utf-8")
            parent = EvaluationReport(0.5, {"passed": 5}, "parent", "parent.log")
            session = CandidateEvaluationSession(
                root,
                repository,
                FakeEvaluator(),
                parent,
                root / "generation",
                [["python", "-m", "py_compile", str(source)]],
                1,
            )

            session.evaluate()
            best = session.restore_best()

            self.assertIsNotNone(best)
            self.assertEqual(source.read_text(encoding="utf-8"), "VALUE = 2\n")
            self.assertTrue(Path(best.patch_path).is_file())
            repository.rollback()
            self.assertEqual(source.read_text(encoding="utf-8"), "VALUE = 1\n")

    @staticmethod
    def _git(root: Path, *arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
