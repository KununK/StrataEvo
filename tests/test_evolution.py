import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strataevo.evolution.attempts import CandidateEvaluationSession
from strataevo.evolution.cli import (
    _prepare_generation_dir,
    _promotion_decision,
    parse_args,
    run_one_generation,
)
from strataevo.evolution.contract import EvaluationContract
from strataevo.evolution.diagnosis import Diagnosis, DiagnosisReport, EvolutionLayer
from strataevo.evolution.git import GitRepository
from strataevo.evolution.mutator import _round_step_budget, _run_refinement_session
from strataevo.evolution.plan import (
    EvolutionPlan,
    EvolutionPlanReport,
    ExpectedOutcome,
    MetricDirection,
)
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from strataevo.evolution.workspace import SelfWorkspace
from tinyagent import AgentResult, Message, Usage

TEST_CONTRACT = EvaluationContract(
    benchmark="test",
    objective="test direct agent behavior",
    direct_paths=("src/tinyagent",),
    deferred_paths=("src/strataevo/evolution/mutator.py",),
)


class EvolutionTests(unittest.TestCase):
    def test_mutator_default_reserves_repair_budget(self):
        args = parse_args([])
        self.assertEqual(args.benchmark, "humaneval")
        self.assertEqual(args.mutator_max_steps, 200)
        self.assertEqual(args.mutator_rounds, 5)
        self.assertEqual(args.max_eval_attempts, 5)
        self.assertEqual(args.benchmark_max_steps, 12)
        self.assertIsNone(args.eval_limit)

    def test_refinement_session_returns_evaluation_feedback_to_same_agent(self):
        class FakeAgent:
            max_steps = 0

            def __init__(self):
                self.prompts = []
                self.budgets = []

            def run(self, prompt, *, session_id):
                self.prompts.append((prompt, session_id))
                self.budgets.append(self.max_steps)
                return AgentResult("", [Message("assistant", "")], Usage(10, 2), 1, "max_steps")

        feedback = iter(
            [
                '{"outcome_type":"validation_failed","reason":"syntax error"}',
                '{"outcome_type":"evaluated","candidate_task_score":0.6}',
            ]
        )
        agent = FakeAgent()
        config = EvolutionConfig(
            repo=".",
            run_name="test",
            mutator_max_steps=10,
            mutator_rounds=2,
            max_eval_attempts=2,
        )

        result = _run_refinement_session(agent, "initial", lambda: next(feedback), config)

        self.assertEqual(len(agent.prompts), 2)
        self.assertEqual(agent.prompts[0], ("initial", "generation-refinement"))
        self.assertIn("validation_failed", agent.prompts[1][0])
        self.assertEqual(agent.budgets, [5, 9])
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.usage, Usage(20, 4))
        self.assertEqual(result.stop_reason, "round_limit")

    def test_round_step_budget_adapts_to_total_and_round_count(self):
        for total, rounds, expected in (
            (200, 5, [40, 40, 40, 40, 40]),
            (200, 3, [67, 67, 66]),
            (10, 4, [3, 3, 2, 2]),
        ):
            remaining = total
            budgets = []
            for index in range(rounds):
                budget = _round_step_budget(remaining, rounds - index)
                budgets.append(budget)
                remaining -= budget
            self.assertEqual(budgets, expected)
            self.assertEqual(sum(budgets), total)

    def test_old_config_uses_default_mutator_rounds(self):
        config = EvolutionConfig.from_dict({"repo": ".", "run_name": "old-run"})
        self.assertEqual(config.mutator_rounds, 5)

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
            tools["replace_lines"].run(
                {
                    "path": "src/tinyagent/new.py",
                    "start_line": 1,
                    "end_line": 1,
                    "content": "VALUE = 2",
                }
            )
            self.assertEqual(
                (root / "src/tinyagent/new.py").read_text(encoding="utf-8"),
                "VALUE = 2\n",
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

    def test_validation_failure_does_not_use_benchmark_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/tinyagent"
            source.mkdir(parents=True)
            agent_file = source / "agent.py"
            agent_file.write_text("VERSION = 0\n", encoding="utf-8")
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "test")
            self._git(root, "config", "user.email", "test@example.com")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "baseline")

            class FakeEvaluator:
                contract = TEST_CONTRACT

                def evaluate(self, _output_dir):
                    return EvaluationReport(0.6, {}, "candidate", "candidate.log")

            session = CandidateEvaluationSession(
                root,
                GitRepository(root, ["src/tinyagent"]),
                FakeEvaluator(),
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                root / "attempts",
                [["validate"]],
                1,
            )
            (root / "attempts").mkdir()
            agent_file.write_text("VERSION = broken\n", encoding="utf-8")
            with patch(
                "strataevo.evolution.attempts.run_commands",
                return_value=(False, "syntax error"),
            ):
                feedback = json.loads(session.evaluate())

            self.assertEqual(feedback["outcome_type"], "validation_failed")
            self.assertEqual(feedback["evaluations_used"], 0)
            self.assertEqual(feedback["evaluations_remaining"], 1)
            self.assertTrue(feedback["candidate_retained"])
            self.assertEqual(feedback["working_tree_state"], "current_candidate")

            duplicate = json.loads(session.evaluate())

            self.assertTrue(duplicate["cached"])
            self.assertFalse(duplicate["candidate_retained"])
            self.assertEqual(duplicate["working_tree_state"], "parent")
            self.assertEqual(agent_file.read_text(encoding="utf-8"), "VERSION = 0\n")
            self.assertIn("submitted patch is not active", duplicate["instruction"])

            agent_file.write_text("VERSION = 1\n", encoding="utf-8")
            with patch(
                "strataevo.evolution.attempts.run_commands",
                return_value=(True, "ok"),
            ):
                feedback = json.loads(session.evaluate())

            self.assertEqual(feedback["outcome_type"], "evaluated")
            self.assertEqual(feedback["evaluations_used"], 1)
            self.assertEqual(feedback["evaluations_remaining"], 0)
            self.assertEqual(len(session.attempts), 2)

    def test_regression_restores_best_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/tinyagent"
            source.mkdir(parents=True)
            agent_file = source / "agent.py"
            agent_file.write_text("VERSION = 0\n", encoding="utf-8")
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "test")
            self._git(root, "config", "user.email", "test@example.com")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "baseline")
            reports = iter(
                [
                    EvaluationReport(0.7, {}, "first", "first.log"),
                    EvaluationReport(0.6, {}, "second", "second.log"),
                ]
            )

            class FakeEvaluator:
                contract = TEST_CONTRACT

                def evaluate(self, _output_dir):
                    return next(reports)

            attempts = root / "attempts"
            attempts.mkdir()
            session = CandidateEvaluationSession(
                root,
                GitRepository(root, ["src/tinyagent"]),
                FakeEvaluator(),
                EvaluationReport(0.5, {}, "parent", "parent.log"),
                attempts,
                [],
                2,
            )
            agent_file.write_text("VERSION = 1\n", encoding="utf-8")
            session.evaluate()
            agent_file.write_text("VERSION = 2\n", encoding="utf-8")
            feedback = json.loads(session.evaluate())

            self.assertIn("restored best candidate", feedback["reason"])
            self.assertFalse(feedback["candidate_retained"])
            self.assertEqual(feedback["working_tree_state"], "best_candidate")
            self.assertEqual(agent_file.read_text(encoding="utf-8"), "VERSION = 1\n")
            self.assertEqual(session.evaluations_used, 2)

    def test_promotion_requires_strictly_higher_task_score(self):
        parent = EvaluationReport(0.8, {}, "parent", "parent.log")
        better_score = EvaluationReport(0.9, {}, "candidate", "candidate.log")
        worse_score = EvaluationReport(0.7, {}, "candidate", "candidate.log")
        equal_score = EvaluationReport(0.8, {}, "candidate", "candidate.log")

        self.assertTrue(_promotion_decision(parent, better_score)[0])
        self.assertFalse(_promotion_decision(parent, worse_score)[0])
        self.assertFalse(_promotion_decision(parent, equal_score)[0])

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

    def test_full_score_stops_before_diagnosis(self):
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

            run_dir = root / "evolution/runs/test"
            run_dir.mkdir(parents=True)
            config = EvolutionConfig(
                repo=str(root),
                run_name="test",
                branch="evo",
                mutable_paths=["src/tinyagent"],
            )
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

            class FullScoreEvaluator:
                contract = TEST_CONTRACT

                def __init__(self, _repo, _config):
                    pass

                def evaluate(self, _output_dir):
                    return EvaluationReport(1.0, {}, "baseline", "baseline.log")

            with (
                patch(
                    "strataevo.evolution.cli.create_evaluator",
                    return_value=FullScoreEvaluator(root, config),
                ),
                patch("strataevo.evolution.cli.diagnose_evaluation") as diagnose,
            ):
                self.assertEqual(run_one_generation(config_path), 0)

            diagnose.assert_not_called()
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertTrue(state["completed"])
            self.assertEqual(state["completion_reason"], "task score reached maximum 1.0")
            self.assertFalse((run_dir / "generation-0001").exists())

    def test_keyboard_interrupt_rolls_back_candidate_changes(self):
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
            commit = self._git_output(root, "rev-parse", "HEAD").strip()

            run_dir = root / "evolution/runs/test"
            run_dir.mkdir(parents=True)
            config = EvolutionConfig(
                repo=str(root),
                run_name="test",
                branch="evo",
                mutable_paths=["src/tinyagent"],
            )
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            report = EvaluationReport(0.5, {}, "parent", "parent.log")
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

            def interrupted_mutation(*_args):
                agent_file.write_text("VERSION = 1\n", encoding="utf-8")
                raise KeyboardInterrupt

            with (
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch(
                    "strataevo.evolution.cli.diagnose_evaluation",
                    return_value=self._diagnosis_report(),
                ),
                patch("strataevo.evolution.cli.plan_evolution", return_value=self._plan_report()),
                patch("strataevo.evolution.cli.mutate", side_effect=interrupted_mutation),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_one_generation(config_path)

            self.assertEqual(agent_file.read_text(encoding="utf-8"), "VERSION = 0\n")
            GitRepository(root, ["src/tinyagent"]).ensure_clean()
            failure = json.loads(
                (run_dir / "generation-0001/failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["error_type"], "KeyboardInterrupt")
            self.assertEqual(failure["error"], "interrupted by user")

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
                    EvaluationReport(0.5, {}, "parent", "parent.log"),
                    EvaluationReport(0.6, {}, "child", "child.log"),
                    EvaluationReport(0.5, {}, "fresh-parent", "fresh-parent.log"),
                    EvaluationReport(0.6, {}, "confirmed-child", "confirmed-child.log"),
                ]
            )

            class FakeEvaluator:
                contract = TEST_CONTRACT

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
            plan = self._plan_report()

            def fake_mutate(
                _repo,
                _config,
                _generation,
                _report,
                _diagnosis,
                _plan,
                _history,
                _directory,
                _commands,
                _contract,
                evaluate_candidate,
            ):
                agent_file.write_text("VERSION = 1\n", encoding="utf-8")
                evaluate_candidate()
                return AgentResult(
                    "updated", [Message("assistant", "updated")], Usage(), 1, "completed"
                )

            with (
                patch(
                    "strataevo.evolution.cli.create_evaluator",
                    return_value=FakeEvaluator(root, config),
                ),
                patch("strataevo.evolution.cli.diagnose_evaluation", return_value=diagnosis),
                patch("strataevo.evolution.cli.plan_evolution", return_value=plan),
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch("strataevo.evolution.cli.mutate", side_effect=fake_mutate),
            ):
                self.assertEqual(run_one_generation(config_path), 0)

            record = json.loads(
                (run_dir / "generation-0001/record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["decision"], "accepted")
            self.assertEqual(record["outcome_type"], "accepted")
            self.assertEqual(record["diagnosed_layers"], ["architecture"])
            self.assertEqual(record["planned_layer"], "architecture")
            memory = self._read_jsonl(run_dir / "evolution_memory.jsonl")
            self.assertEqual(len(memory), 1)
            self.assertEqual(memory[0]["generation"], 1)
            self.assertEqual(memory[0]["decision"], "accepted")
            self.assertNotIn("utility_delta", memory[0])
            self.assertTrue(memory[0]["patch_path"].endswith("changes.patch"))
            self.assertIn("VERSION = 1", memory[0]["patch_excerpt"])
            self.assertTrue(memory[0]["outcome_observations"][0]["satisfied"])
            self.assertEqual(memory[0]["outcome"]["status"], "supported")
            self.assertAlmostEqual(memory[0]["outcome"]["score_delta"], 0.1)
            self.assertEqual(agent_file.read_text(encoding="utf-8"), "VERSION = 1\n")
            commit_subject = self._git_output(root, "log", "-1", "--pretty=%s").strip()
            self.assertEqual(commit_subject, "evolve: generation 1 pass@1 0.600000")

    def test_generation_commits_best_evaluated_attempt(self):
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
            run_dir.mkdir(parents=True)
            config = EvolutionConfig(
                repo=str(root),
                run_name="test",
                branch="evo",
                mutable_paths=["src/tinyagent"],
                max_eval_attempts=2,
            )
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            reports = iter(
                [
                    EvaluationReport(0.5, {}, "parent", "parent.log"),
                    EvaluationReport(0.7, {}, "first", "first.log"),
                    EvaluationReport(0.6, {}, "second", "second.log"),
                    EvaluationReport(0.5, {}, "fresh-parent", "fresh-parent.log"),
                    EvaluationReport(0.65, {}, "confirmed-child", "confirmed-child.log"),
                ]
            )

            class FakeEvaluator:
                contract = TEST_CONTRACT

                def __init__(self, _repo, _config):
                    pass

                def evaluate(self, _output_dir):
                    return next(reports)

            def fake_mutate(*args):
                evaluate_candidate = args[-1]
                agent_file.write_text("VERSION = 1\n", encoding="utf-8")
                evaluate_candidate()
                agent_file.write_text("VERSION = 2\n", encoding="utf-8")
                evaluate_candidate()
                agent_file.write_text("VERSION = 3\n", encoding="utf-8")
                return AgentResult(
                    "tested twice", [Message("assistant", "done")], Usage(), 4, "completed"
                )

            with (
                patch(
                    "strataevo.evolution.cli.create_evaluator",
                    return_value=FakeEvaluator(root, config),
                ),
                patch(
                    "strataevo.evolution.cli.diagnose_evaluation",
                    return_value=self._diagnosis_report(),
                ),
                patch(
                    "strataevo.evolution.cli.plan_evolution",
                    return_value=self._plan_report(),
                ),
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch("strataevo.evolution.cli.mutate", side_effect=fake_mutate),
            ):
                self.assertEqual(run_one_generation(config_path), 0)

            self.assertEqual(agent_file.read_text(encoding="utf-8"), "VERSION = 1\n")
            record = json.loads(
                (run_dir / "generation-0001/record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["candidate_report"]["task_score"], 0.65)
            self.assertEqual(record["promotion_parent_report"]["task_score"], 0.5)
            self.assertEqual(len(record["evaluation_attempts"]), 2)
            self.assertEqual(
                [item["report"]["task_score"] for item in record["evaluation_attempts"]],
                [0.7, 0.6],
            )

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
            report = EvaluationReport(0.5, {}, "parent", "parent.log")
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
            plan = self._plan_report()
            with (
                patch("strataevo.evolution.cli.validation_commands", return_value=[]),
                patch("strataevo.evolution.cli.diagnose_evaluation", return_value=diagnosis),
                patch("strataevo.evolution.cli.plan_evolution", return_value=plan),
                patch("strataevo.evolution.cli.mutate", return_value=agent_result),
            ):
                self.assertEqual(run_one_generation(run_dir / "config.json"), 0)

            archived = run_dir / "generation-0001-failed-0001/failure.json"
            self.assertTrue(archived.is_file())
            record = json.loads(
                (run_dir / "generation-0001/record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["decision"], "rejected")
            self.assertEqual(record["outcome_type"], "no_change")
            memory = self._read_jsonl(run_dir / "evolution_memory.jsonl")
            self.assertEqual(len(memory), 1)
            self.assertEqual(memory[0]["decision"], "rejected")
            self.assertIsNone(memory[0]["candidate_task_score"])

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

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def _diagnosis_report() -> DiagnosisReport:
        return DiagnosisReport(
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

    @staticmethod
    def _plan_report() -> EvolutionPlanReport:
        return EvolutionPlanReport(
            plan=EvolutionPlan(
                target_diagnosis=0,
                primary_layer=EvolutionLayer.ARCHITECTURE,
                hypothesis="test hypothesis",
                intervention="test intervention",
                expected_outcomes=[
                    ExpectedOutcome(
                        metric="task_score",
                        direction=MetricDirection.NON_DECREASING,
                        reason="preserve task quality",
                    )
                ],
                likely_files=["src/tinyagent/agent.py"],
                expected_long_term_value="improve later agent control flow",
                prerequisites=[],
                confidence=0.8,
            ),
            available_metrics={"task_score": 0.5},
            input_tokens=0,
            output_tokens=0,
            raw_output="{}",
            attempts=["{}"],
        )


if __name__ == "__main__":
    unittest.main()
