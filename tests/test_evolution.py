import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strataevo.evolution.cli import _prepare_generation_dir, _promotion_decision, run_one_generation
from strataevo.evolution.diagnosis import Diagnosis, DiagnosisReport, EvolutionLayer
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

    def test_incomplete_generation_is_archived_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory) / "generation-0001"
            generation.mkdir()
            (generation / "failure.json").write_text('{"error":"invalid JSON"}\n', encoding="utf-8")

            _prepare_generation_dir(generation)

            archive = Path(directory) / "generation-0001-failed-0001"
            self.assertTrue((archive / "failure.json").is_file())
            self.assertTrue(generation.is_dir())
            self.assertEqual(list(generation.iterdir()), [])

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

            diagnosis = DiagnosisReport(
                source_dir="parent",
                input_case_count=1,
                diagnoses=[
                    Diagnosis(
                        primary_layer=EvolutionLayer.ARCHITECTURE,
                        related_layers=[],
                        problem="test problem",
                        evidence=["test evidence"],
                        affected_tasks=["test/1"],
                        proposed_direction="test direction",
                        confidence=1.0,
                    )
                ],
                input_tokens=0,
                output_tokens=0,
                raw_output="{}",
                attempts=["{}"],
            )

            def fake_mutate(
                _repo, _config, _generation, _report, _diagnosis, _directory, _commands
            ):
                agent_file.write_text("VERSION = 1\n", encoding="utf-8")
                return AgentResult(
                    "updated", [Message("assistant", "updated")], Usage(), 1, "completed"
                )

            with (
                patch("strataevo.evolution.cli.HumanEvalEvaluator", FakeEvaluator),
                patch("strataevo.evolution.cli.diagnose_evaluation", return_value=diagnosis),
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch("strataevo.evolution.cli.mutate", side_effect=fake_mutate),
            ):
                self.assertEqual(run_one_generation(config_path), 0)

            record = json.loads(
                (run_dir / "generation-0001/record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["decision"], "accepted")
            self.assertEqual(record["diagnosed_layers"], ["architecture"])
            self.assertEqual(agent_file.read_text(encoding="utf-8"), "VERSION = 1\n")
            commit_subject = self._git_output(root, "log", "-1", "--pretty=%s")
            self.assertIn("evolve: generation 1", commit_subject)

    def test_failed_diagnosis_can_resume_same_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/tinyagent"
            source.mkdir(parents=True)
            (source / "agent.py").write_text("VERSION = 0\n", encoding="utf-8")
            (root / ".gitignore").write_text("evolution/runs/\n", encoding="utf-8")
            self._git(root, "init", "-b", "evo")
            self._git(root, "config", "user.name", "test")
            self._git(root, "config", "user.email", "test@example.com")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "baseline")
            commit = self._git_output(root, "rev-parse", "HEAD").strip()

            run_dir = root / "evolution/runs/test"
            run_dir.mkdir(parents=True)
            config = EvolutionConfig(repo=str(root), run_name="test", branch="evo")
            (run_dir / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
            report = EvaluationReport(0.5, 0.49, {}, "parent", "parent.log")
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "next_generation": 1,
                        "current_commit": commit,
                        "current_report": report.to_dict(),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch(
                    "strataevo.evolution.cli.diagnose_evaluation",
                    side_effect=ValueError("invalid diagnosis JSON"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "invalid diagnosis JSON"):
                    run_one_generation(run_dir / "config.json")

            failure = run_dir / "generation-0001/failure.json"
            self.assertTrue(failure.is_file())

            diagnosis = DiagnosisReport(
                source_dir="parent",
                input_case_count=1,
                diagnoses=[
                    Diagnosis(
                        primary_layer=EvolutionLayer.ARCHITECTURE,
                        related_layers=[],
                        problem="test problem",
                        evidence=["test evidence"],
                        affected_tasks=["test/1"],
                        proposed_direction="test direction",
                        confidence=1.0,
                    )
                ],
                input_tokens=0,
                output_tokens=0,
                raw_output="{}",
                attempts=["{}"],
            )
            agent_result = AgentResult(
                "no change", [Message("assistant", "no change")], Usage(), 1, "completed"
            )
            with (
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch("strataevo.evolution.cli.diagnose_evaluation", return_value=diagnosis),
                patch("strataevo.evolution.cli.mutate", return_value=agent_result),
            ):
                self.assertEqual(run_one_generation(run_dir / "config.json"), 0)

            archived = run_dir / "generation-0001-failed-0001/failure.json"
            self.assertTrue(archived.is_file())
            record = json.loads(
                (run_dir / "generation-0001/record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["decision"], "rejected")

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
