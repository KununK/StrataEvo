import argparse
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.generalization import evaluate, export, freeze, load_snapshot


class FrozenStateTests(unittest.TestCase):
    def test_source_archive_and_context_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "src/tinyagent/agent.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                            "-c", "user.email=test@example.com", "commit", "-qm", "source"],
                           check=True)
            state = root / "state.json"
            context = {"task_prompt_addendum": "Keep the final artifact"}
            state.write_text(json.dumps({"current_context": context}))
            freeze(root, root / "frozen.json", state, "base")
            export(root, root / "frozen.json", root / "bundle")
            with tarfile.open(root / "bundle/source.tar") as archive:
                self.assertEqual(archive.extractfile("src/tinyagent/agent.py").read(),
                                 source.read_bytes())
            self.assertEqual(load_snapshot(root / "bundle/frozen_state.json")["context"], context)

    def test_evaluate_loads_all_profiles_and_releases_adapter_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({
                "commit": "source", "model": "old-alias", "base_model": "base",
                "adapter": {"path": "adapter"},
                "context": {"task_prompt_addendum": "Verify"},
                "tools": {"addenda": {"run_shell": "Preserve outputs"}},
            }))
            args = argparse.Namespace(base_url="http://localhost:8000/v1", workers=2,
                                      max_steps=12, benchmarks=["mbpp"])
            with patch("eval.generalization.subprocess.run") as run, \
                    patch("eval.generalization.VLLMAdapterRuntime") as runtime:
                run.side_effect = [None, subprocess.CalledProcessError(1, "benchmark")]
                with self.assertRaises(subprocess.CalledProcessError):
                    evaluate(root, frozen, root / "evaluation", args)
                command = run.call_args.args[0]
                self.assertIn("--context-file", command)
                self.assertIn("--tool-profile-file", command)
                self.assertEqual(command[command.index("--model") + 1],
                                 "generalization-evaluation")
                runtime.return_value.activate.assert_called_once_with(
                    "generalization-evaluation", root / "adapter")
                runtime.return_value.deactivate.assert_called_once_with(
                    "generalization-evaluation")

    def test_bundle_adapter_path_is_relative_to_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen_state.json"
            frozen.write_text(json.dumps({"adapter": {"path": "adapter"}}))
            snapshot = load_snapshot(frozen)
            self.assertEqual(snapshot["adapter"]["path"], str(root / "adapter"))
            self.assertEqual(json.loads(frozen.read_text())["adapter"]["path"], "adapter")

    def test_complete_state_is_frozen_without_reports_or_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            state.write_text(json.dumps({
                "current_commit": "old-source", "current_model": "base-model",
                "current_adapter": None, "current_context": {"task_prompt_addendum": "Verify"},
                "current_tool_profile": {"addenda": {"run_shell": "Keep outputs"}},
                "current_report": {"metrics": {"passed": 123}},
            }))
            with patch("eval.generalization.git", return_value="resolved-commit"):
                snapshot = freeze(root, root / "frozen.json", state, "fallback")
            self.assertEqual(snapshot["commit"], "resolved-commit")
            self.assertEqual(snapshot["base_model"], "fallback")
            self.assertEqual(snapshot["context"], {"task_prompt_addendum": "Verify"})
            self.assertIsNotNone(snapshot["tools"])
            self.assertNotIn("current_report", snapshot)
            with self.assertRaises(FileExistsError):
                freeze(root, root / "frozen.json", state, "fallback")

    def test_missing_adapter_is_not_silently_evaluated_as_base_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            state.write_text(json.dumps({"current_adapter": {"path": str(root / "missing")}}))
            with patch("eval.generalization.git", return_value="resolved-commit"):
                with self.assertRaises(FileNotFoundError):
                    freeze(root, root / "frozen.json", state, "base")

    def test_export_copies_adapter_and_removes_local_profile_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "weights"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "base-model"}))
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({
                "commit": "source-commit", "source_state": "/old/state.json",
                "adapter": {"name": "trained", "path": str(adapter)},
                "context": {"path": "/old/context.json", "task_prompt_addendum": "Verify"},
                "tools": None,
            }))
            with patch("eval.generalization.subprocess.run"):
                export(root, frozen, root / "bundle")
            data = json.loads((root / "bundle/frozen_state.json").read_text())
            self.assertEqual(data["adapter"]["path"], "adapter")
            self.assertEqual(data["base_model"], "base-model")
            self.assertNotIn("path", data["context"])
            self.assertTrue((root / "bundle/adapter/adapter_config.json").is_file())
            self.assertEqual(json.loads(frozen.read_text())["source_state"], "/old/state.json")
