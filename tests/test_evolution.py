import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strataevo.evolution.cli import _promotion_decision, run_one_generation
from strataevo.evolution.git import GitRepository
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from strataevo.evolution.workspace import SelfWorkspace
from tinyagent import AgentResult, Message, Usage


class EvolutionTests(unittest.TestCase):
    def test_self_workspace_can_only_write_evolvable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/tinyagent").mkdir(parents=True)
            (root / "tests").mkdir()
            workspace = SelfWorkspace(root, ["src/tinyagent"], [])
            tools = {item.name: item for item in workspace.tools()}

            tools["write_file"].run({"path": "src/tinyagent/new.py", "content": "VALUE = 1\n"})
            self.assertEqual(
                (root / "src/tinyagent/new.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )
            with self.assertRaises(PermissionError):
                tools["write_file"].run({"path": "tests/test_backdoor.py", "content": "pass\n"})

    def test_git_repository_rolls_back_tracked_and_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/tinyagent"
            source.mkdir(parents=True)
            existing = source / "agent.py"
            existing.write_text("OLD = True\n", encoding="utf-8")
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "test")
            self._git(root, "config", "user.email", "test@example.com")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "baseline")

            existing.write_text("OLD = False\n", encoding="utf-8")
            added = source / "new.py"
            added.write_text("NEW = True\n", encoding="utf-8")
            repository = GitRepository(root, ["src/tinyagent"])
            self.assertEqual(
                repository.changed_paths(),
                ["src/tinyagent/agent.py", "src/tinyagent/new.py"],
            )
            repository.stage()
            self.assertIn("src/tinyagent/new.py", repository.changed_paths())
            repository.rollback()

            self.assertEqual(existing.read_text(encoding="utf-8"), "OLD = True\n")
            self.assertFalse(added.exists())
            repository.ensure_clean()

    def test_promotion_requires_preserved_score_and_higher_utility(self):
        config = EvolutionConfig(repo=".", run_name="test")
        parent = EvaluationReport(0.8, 0.79, {}, "parent", "parent.log")
        better = EvaluationReport(0.8, 0.795, {}, "candidate", "candidate.log")
        worse_score = EvaluationReport(0.7, 0.8, {}, "candidate", "candidate.log")
        equal = EvaluationReport(0.8, 0.79, {}, "candidate", "candidate.log")

        self.assertTrue(_promotion_decision(parent, better, config)[0])
        self.assertFalse(_promotion_decision(parent, worse_score, config)[0])
        self.assertFalse(_promotion_decision(parent, equal, config)[0])

    def test_one_generation_commits_an_improved_self_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/tinyagent"
            source.mkdir(parents=True)
            agent_file = source / "agent.py"
            agent_file.write_text("VERSION = 0\n", encoding="utf-8")
            (root / ".gitignore").write_text("evolution/runs/\n", encoding="utf-8")
            self._git(root, "init", "-b", "evo")
            self._git(root, "config", "user.name", "test")
            self._git(root, "config", "user.email", "test@example.com")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "baseline")

            run_dir = root / "evolution/runs/test"
            config_path = run_dir / "config.json"
            config = EvolutionConfig(
                repo=str(root),
                run_name="test",
                branch="evo",
                mutable_paths=["src/tinyagent"],
            )
            run_dir.mkdir(parents=True)
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

            reports = iter(
                [
                    EvaluationReport(0.5, 0.49, {}, "parent", "parent.log"),
                    EvaluationReport(0.5, 0.495, {}, "child", "child.log"),
                ]
            )

            class FakeEvaluator:
                def __init__(self, _repo, _config):
                    pass

                def evaluate(self, _output_dir):
                    return next(reports)

            def fake_mutate(_repo, _config, _generation, _report, _directory, _commands):
                agent_file.write_text("VERSION = 1\n", encoding="utf-8")
                return AgentResult(
                    "updated", [Message("assistant", "updated")], Usage(), 1, "completed"
                )

            with (
                patch("strataevo.evolution.cli.HumanEvalEvaluator", FakeEvaluator),
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch("strataevo.evolution.cli.mutate", side_effect=fake_mutate),
            ):
                self.assertEqual(run_one_generation(config_path), 0)

            record = json.loads(
                (run_dir / "generation-0001/record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["decision"], "accepted")
            self.assertEqual(agent_file.read_text(encoding="utf-8"), "VERSION = 1\n")
            commit_subject = self._git_output(root, "log", "-1", "--pretty=%s")
            self.assertIn("evolve: generation 1", commit_subject)

    @staticmethod
    def _git(root: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _git_output(root: Path, *arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=root, text=True)


if __name__ == "__main__":
    unittest.main()
