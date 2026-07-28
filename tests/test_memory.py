import json
import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.memory import (
    EvolutionMemory,
    EvolutionMemoryEntry,
    MemoryDiagnosis,
    MemoryOutcome,
    _task_transitions,
    memory_context,
)


class EvolutionMemoryTests(unittest.TestCase):
    def test_append_is_persistent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EvolutionMemory(Path(directory) / "evolution_memory.jsonl")
            entry = self._entry(1, "context", "rejected")

            memory.append(entry)
            memory.append(entry)

            self.assertEqual(memory.load(), [entry])
            self.assertEqual(len(memory.path.read_text(encoding="utf-8").splitlines()), 1)

    def test_conflicting_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EvolutionMemory(Path(directory) / "evolution_memory.jsonl")
            memory.append(self._entry(1, "context", "rejected"))

            with self.assertRaisesRegex(ValueError, "conflicting"):
                memory.append(self._entry(1, "context", "accepted"))

    def test_relevant_history_prioritizes_matching_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EvolutionMemory(Path(directory) / "evolution_memory.jsonl")
            memory.append(self._entry(1, "tools", "rejected"))
            memory.append(self._entry(2, "context", "accepted"))
            memory.append(self._entry(3, "architecture", "rejected", related=["tools"]))

            selected = memory.relevant({"tools"}, limit=2)

            self.assertEqual([entry.generation for entry in selected], [1, 3])

    def test_contract_mismatches_remain_loadable_evidence(self):
        for outcome in (
            "deferred_change",
            "mixed_change_scope",
            "unclassified_change",
        ):
            with self.subTest(outcome=outcome):
                data = self._entry(1, "architecture", "rejected").to_dict()
                data["outcome_type"] = outcome

                loaded = EvolutionMemoryEntry.from_dict(data)

                self.assertEqual(loaded.outcome_type, outcome)

    def test_memory_context_budget_keeps_the_newest_entries(self):
        entries = [self._entry(number, "context", "rejected") for number in range(1, 4)]
        newest_size = len(json.dumps(entries[-1].to_context_dict())) + 3

        context = memory_context(entries, max_chars=newest_size)

        self.assertEqual([item["generation"] for item in context], [3])

    def test_context_exposes_problem_action_and_outcome(self):
        entry = self._entry(1, "tools", "rejected")
        entry.outcome = MemoryOutcome(
            status="regressed",
            score_delta=-0.1,
            fixed_tasks=["task/1"],
            regressed_tasks=["task/2"],
            remaining_failures=[f"task/{number}" for number in range(20)],
            summary="The intervention regressed.",
            next_step="Revise the hypothesis.",
        )

        context = entry.to_context_dict()

        self.assertEqual(context["selected_diagnosis"]["problem"], "test problem")
        self.assertEqual(context["action"]["primary_layer"], "tools")
        self.assertEqual(context["outcome"]["status"], "regressed")
        self.assertEqual(context["outcome"]["remaining_failures"]["count"], 20)
        self.assertEqual(len(context["outcome"]["remaining_failures"]["examples"]), 12)
        self.assertNotIn("agent_output", context)

    def test_old_entry_derives_outcome_summary(self):
        data = self._entry(1, "context", "rejected").to_dict()
        data.pop("outcome")
        data["parent_task_score"] = 0.6
        data["candidate_task_score"] = 0.5

        loaded = EvolutionMemoryEntry.from_dict(data)

        self.assertEqual(loaded.outcome.status, "regressed")
        self.assertAlmostEqual(loaded.outcome.score_delta or 0.0, -0.1)

    def test_task_transitions_are_read_from_generic_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            candidate = root / "candidate"
            parent.mkdir()
            candidate.mkdir()
            (parent / "results.jsonl").write_text(
                '{"task_id":"a","passed":false}\n'
                '{"task_id":"b","passed":true}\n'
                '{"task_id":"c","passed":false}\n',
                encoding="utf-8",
            )
            (candidate / "results.jsonl").write_text(
                '{"task_id":"a","passed":true}\n'
                '{"task_id":"b","passed":false}\n'
                '{"task_id":"c","passed":false}\n',
                encoding="utf-8",
            )

            transitions = _task_transitions(
                {"output_dir": str(parent)},
                {"output_dir": str(candidate)},
            )

            self.assertEqual(transitions["fixed_tasks"], ["a"])
            self.assertEqual(transitions["regressed_tasks"], ["b"])
            self.assertEqual(transitions["remaining_failures"], ["b", "c"])

    @staticmethod
    def _entry(
        generation: int,
        layer: str,
        decision: str,
        *,
        related: list[str] | None = None,
    ) -> EvolutionMemoryEntry:
        return EvolutionMemoryEntry(
            generation=generation,
            parent_commit=f"parent-{generation}",
            resulting_commit=f"child-{generation}" if decision == "accepted" else None,
            decision=decision,
            outcome_type="accepted" if decision == "accepted" else "benchmark_rejected",
            reason="test outcome",
            diagnoses=[
                MemoryDiagnosis(
                    primary_layer=layer,
                    related_layers=related or [],
                    problem="test problem",
                    affected_tasks=["HumanEval/0"],
                    proposed_direction="test direction",
                    confidence=0.8,
                )
            ],
            plan={
                "target_diagnosis": 0,
                "primary_layer": layer,
                "hypothesis": "test hypothesis",
                "intervention": "test intervention",
            },
            outcome_observations=[],
            changed_paths=["src/tinyagent/agent.py"],
            patch_path=f"generation-{generation:04d}/changes.patch",
            patch_excerpt="- old behavior\n+ new behavior",
            agent_output="updated implementation",
            parent_task_score=0.5,
            candidate_task_score=0.5,
        )


if __name__ == "__main__":
    unittest.main()
