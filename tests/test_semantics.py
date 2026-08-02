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

    def test_python_module_docstring_only_changes_are_semantic_noop(self):
        parent = '''"""Old module documentation."""

class Agent:
    def run(self):
        return 1
'''
        candidate_source = '''"""Expanded module documentation."""

class Agent:
    def run(self):
        return 1
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "agent.py"
            candidate.write_text(candidate_source, encoding="utf-8")

            result = classify_changes(root, ["agent.py"], lambda _path: parent)

        self.assertEqual(result.classification, "semantic_noop")

    def test_python_function_docstring_change_is_not_noop(self):
        parent = 'def run():\n    """Old tool description."""\n    return 1\n'
        candidate_source = 'def run():\n    """New tool description."""\n    return 1\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "agent.py"
            candidate.write_text(candidate_source, encoding="utf-8")

            result = classify_changes(root, ["agent.py"], lambda _path: parent)

        self.assertEqual(result.classification, "behavior_change")

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
