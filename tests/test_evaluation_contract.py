import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.attempts import CandidateEvaluationSession
from strataevo.evolution.contract import EvaluationContract
from strataevo.evolution.evaluation import promotion_decision
from strataevo.evolution.git import GitRepository
from strataevo.evolution.types import EvaluationReport

CONTRACT = EvaluationContract(
    benchmark="test",
    objective="measure the task agent",
    direct_paths=("src/tinyagent",),
    deferred_paths=("src/strataevo/evolution/mutator.py",),
)


class EvaluationContractTests(unittest.TestCase):
    def test_classifies_direct_deferred_and_unknown_paths(self):
        impact = CONTRACT.classify(
            [
                "src/tinyagent/agent.py",
                "src/strataevo/evolution/mutator.py",
                "README.md",
            ]
        )

        self.assertEqual(impact.direct_paths, ("src/tinyagent/agent.py",))
        self.assertEqual(impact.deferred_paths, ("src/strataevo/evolution/mutator.py",))
        self.assertEqual(impact.unclassified_paths, ("README.md",))
        self.assertFalse(impact.is_direct_only)

    def test_deferred_change_is_restored_without_running_benchmark(self):
        with self._repository() as (root, repository, evaluator):
            mutator = root / "src/strataevo/evolution/mutator.py"
            mutator.write_text("VERSION = 1\n", encoding="utf-8")
            session = self._session(root, repository, evaluator)

            feedback = json.loads(session.evaluate())

            self.assertEqual(feedback["outcome_type"], "deferred_change")
            self.assertEqual(feedback["evaluations_used"], 0)
            self.assertEqual(evaluator.calls, 0)
            self.assertFalse(feedback["candidate_retained"])
            self.assertEqual(feedback["working_tree_state"], "parent")
            self.assertEqual(mutator.read_text(encoding="utf-8"), "VERSION = 0\n")
            self.assertEqual(
                feedback["change_impact"]["deferred_paths"],
                ["src/strataevo/evolution/mutator.py"],
            )

    def test_mixed_change_is_not_attributed_to_direct_benchmark(self):
        with self._repository() as (root, repository, evaluator):
            (root / "src/tinyagent/agent.py").write_text("VERSION = 1\n", encoding="utf-8")
            (root / "src/strataevo/evolution/mutator.py").write_text(
                "VERSION = 1\n", encoding="utf-8"
            )
            session = self._session(root, repository, evaluator)

            feedback = json.loads(session.evaluate())

            self.assertEqual(feedback["outcome_type"], "mixed_change_scope")
            self.assertEqual(feedback["evaluations_used"], 0)
            self.assertEqual(evaluator.calls, 0)
            self.assertFalse(feedback["candidate_retained"])
            self.assertEqual(feedback["working_tree_state"], "parent")
            self.assertEqual(repository.changed_paths(), [])

    def test_selected_candidate_uses_its_single_screening_report(self):
        reports = iter([EvaluationReport(0.9, {}, "selection", "selection.log")])
        with self._repository(reports) as (root, repository, evaluator):
            agent = root / "src/tinyagent/agent.py"
            agent.write_text("VERSION = 1\n", encoding="utf-8")
            session = self._session(root, repository, evaluator)
            session.evaluate()
            best = session.restore_best()
            assert best is not None

            candidate = EvaluationReport.from_dict(best.report)

            self.assertTrue(promotion_decision(session.parent_report, candidate)[0])
            self.assertEqual(agent.read_text(encoding="utf-8"), "VERSION = 1\n")
            self.assertFalse((root / "attempts/promotion").exists())

    @staticmethod
    def _session(root, repository, evaluator):
        attempts = root / "attempts"
        attempts.mkdir()
        return CandidateEvaluationSession(
            root,
            repository,
            evaluator,
            EvaluationReport(0.5, {}, "parent", "parent.log"),
            attempts,
            [],
            5,
        )

    @staticmethod
    def _repository(reports=None):
        class RepositoryContext:
            def __enter__(self):
                self.directory = tempfile.TemporaryDirectory()
                root = Path(self.directory.name)
                (root / "src/tinyagent").mkdir(parents=True)
                (root / "src/strataevo/evolution").mkdir(parents=True)
                (root / "src/tinyagent/agent.py").write_text("VERSION = 0\n", encoding="utf-8")
                (root / "src/strataevo/evolution/mutator.py").write_text(
                    "VERSION = 0\n", encoding="utf-8"
                )
                for command in (
                    ["git", "init", "-b", "main"],
                    ["git", "config", "user.name", "test"],
                    ["git", "config", "user.email", "test@example.com"],
                    ["git", "add", "."],
                    ["git", "commit", "-m", "baseline"],
                ):
                    subprocess.run(
                        command,
                        cwd=root,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                class FakeEvaluator:
                    contract = CONTRACT
                    calls = 0

                    def evaluate(inner_self, _output_dir):
                        inner_self.calls += 1
                        if reports is None:
                            return EvaluationReport(0.6, {}, "candidate", "candidate.log")
                        return next(reports)

                mutable = [
                    "src/tinyagent",
                    "src/strataevo/evolution/mutator.py",
                ]
                return root, GitRepository(root, mutable), FakeEvaluator()

            def __exit__(self, *_args):
                self.directory.cleanup()

        return RepositoryContext()


if __name__ == "__main__":
    unittest.main()
