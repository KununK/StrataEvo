import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.semantics import classify_changes


class ChangeSemanticsTests(unittest.TestCase):
    def test_python_comments_and_formatting_are_semantic_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "agent.py"
            candidate.write_text("# explanation\nVALUE=1\n", encoding="utf-8")

            result = classify_changes(
                root,
                ["agent.py"],
                lambda _path: "VALUE = 1\n",
            )

            self.assertEqual(result.classification, "semantic_noop")
            self.assertEqual(result.paths, {"agent.py": "equivalent"})

    def test_behavioral_python_change_is_not_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "agent.py"
            candidate.write_text("VALUE = 2\n", encoding="utf-8")

            result = classify_changes(
                root,
                ["agent.py"],
                lambda _path: "VALUE = 1\n",
            )

            self.assertEqual(result.classification, "behavior_change")


if __name__ == "__main__":
    unittest.main()
