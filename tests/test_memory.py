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

    @staticmethod
    def _entry(
        generation: int,
        layer: str,
        decision: str,
        *,
        related: list[str] | None = None,
    ) -> EvolutionMemoryEntry:
        candidate_utility = 0.6 if decision == "accepted" else 0.4
        return EvolutionMemoryEntry(
            generation=generation,
            parent_commit=f"parent-{generation}",
            resulting_commit=f"child-{generation}" if decision == "accepted" else None,
            decision=decision,
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
            parent_utility=0.49,
            candidate_task_score=0.5,
            candidate_utility=candidate_utility,
            utility_delta=candidate_utility - 0.49,
        )


if __name__ == "__main__":
    unittest.main()
