import json
import tempfile
import unittest
from pathlib import Path

from strataevo.evolution.memory import (
    EvolutionMemoryEntry,
    MemoryDiagnosis,
    MemoryOutcome,
)
from strataevo.evolution.run_summary import write_run_summary


class RunSummaryTests(unittest.TestCase):
    def test_writes_machine_and_human_readable_summaries(self):
        entry = EvolutionMemoryEntry(
            generation=1,
            parent_commit="parent",
            resulting_commit=None,
            decision="rejected",
            outcome_type="semantic_noop",
            reason="no semantic change",
            diagnoses=[
                MemoryDiagnosis(
                    "architecture",
                    [],
                    "problem",
                    ["task/1"],
                    "direction",
                    0.8,
                )
            ],
            plan={
                "target_diagnosis": 0,
                "primary_layer": "architecture",
                "hypothesis": "test hypothesis",
                "intervention": "test intervention",
            },
            outcome_observations=[],
            changed_paths=["src/tinyagent/agent.py"],
            patch_path="changes.patch",
            patch_excerpt="+ # comment",
            agent_output="done",
            parent_task_score=0.5,
            candidate_task_score=None,
            causal_trace={
                "failure_mechanism": "artifact deleted after validation",
                "target_component": ["src/tinyagent/agent.py"],
                "executed_change": {
                    "type": "source_patch",
                    "changed_paths": ["src/tinyagent/agent.py"],
                },
                "alignment": {"status": "layer_aligned", "scope": "layer_only"},
                "observed_effect": {
                    "promotion": {"passed_gain": 1, "required_pass_gain": 2}
                },
            },
            evaluation_attempts=[{"outcome_type": "semantic_noop"}],
            outcome=MemoryOutcome(
                "not_evaluated",
                None,
                [],
                [],
                [],
                "No behavior changed.",
                "Change behavior.",
                "untested",
                [],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "test-run"

            write_run_summary(
                run_dir,
                [entry],
                {
                    "baseline_task_score": 0.4,
                    "current_report": {"task_score": 0.5},
                },
            )

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["baseline_task_score"], 0.4)
            self.assertEqual(summary["diagnosed_layers"], {"architecture": 1})
            self.assertEqual(summary["planned_layers"], {"architecture": 1})
            self.assertEqual(summary["generations"][0]["candidate_type"], "source_patch")
            self.assertEqual(summary["outcomes"], {"semantic_noop": 1})
            self.assertEqual(summary["generations"][0]["semantic_noop_attempts"], 1)
            self.assertEqual(summary["generations"][0]["intervention"], "test intervention")
            self.assertEqual(summary["causal_alignments"], {"layer_aligned": 1})
            self.assertEqual(
                summary["generations"][0]["causal_trace"]["failure_mechanism"],
                "artifact deleted after validation",
            )
            self.assertEqual(
                summary["generations"][0]["intervention_verdict"],
                "untested",
            )
            self.assertIn("test hypothesis", (run_dir / "summary.md").read_text())
            self.assertIn("test intervention", (run_dir / "summary.md").read_text())
            self.assertIn("artifact deleted after validation", (run_dir / "summary.md").read_text())
            self.assertIn("passed gain 1/2", (run_dir / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
