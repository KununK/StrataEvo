import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.memory import EvolutionMemory, EvolutionMemoryEntry, MemoryDiagnosis


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
