import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.memory import EvolutionMemory, EvolutionMemoryEntry


class MemoryTests(unittest.TestCase):
    def test_memory_is_append_only_and_selects_relevant_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EvolutionMemory(Path(directory) / "memory.jsonl")
            first = EvolutionMemoryEntry(
                1,
                "tools",
                "cause",
                "change description",
                "rejected",
                "benchmark_rejected",
                "no gain",
                0.5,
                0.5,
                [],
            )
            second = EvolutionMemoryEntry(
                2,
                "architecture",
                "cause",
                "change loop",
                "accepted",
                "accepted",
                "gain",
                0.5,
                0.6,
                ["src/tinyagent/agent.py"],
            )
            memory.append(first)
            memory.append(second)

            self.assertEqual(memory.load(), [first, second])
            self.assertEqual(memory.relevant({"tools"}), [first])
            with self.assertRaisesRegex(ValueError, "already contains"):
                memory.append(first)


if __name__ == "__main__":
    unittest.main()
