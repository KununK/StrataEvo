import json
import unittest

from strataevo.evolution.decision import EvolutionDecider, EvolutionDecision, EvolutionLayer
from strataevo.evolution.evidence import EvidenceBundle, TaskEvidence, ToolEvent
from strataevo.evolution.memory import EvolutionMemoryEntry
from strataevo.evolution.runtime.structured import StructuredResponseError
from strataevo.evolution.types import EvaluationReport, EvolutionConfig
from tinyagent import Message, ScriptedModel


class DecisionTests(unittest.TestCase):
    def test_decision_uses_ordered_events_and_names_mutable_files(self):
        response = {
            "layer": "architecture",
            "evidence": ["task/1 removed its output after writing it"],
            "affected_tasks": ["task/1"],
            "hypothesis": "The loop permits destructive cleanup after producing the output.",
            "intervention": "Stop after the verified output is produced.",
            "likely_files": ["src/tinyagent/agent.py"],
        }
        model = ScriptedModel([Message("assistant", json.dumps(response))])
        decision = EvolutionDecider(model).decide(
            self._bundle(),
            EvaluationReport(0.5, {}, "evaluation", "evaluation.log"),
            [],
            EvolutionConfig(repo=".", run_name="test"),
        )
        self.assertEqual(decision.layer, EvolutionLayer.ARCHITECTURE)
        system_prompt = model.requests[0][0].content
        request = model.requests[0][1].content
        self.assertIn("executor can change the observed cause", system_prompt)
        self.assertIn("generated-code correctness", system_prompt)
        self.assertIn("operations inside generated code are not agent", system_prompt)
        self.assertIn("assertion failure is model evidence", system_prompt)
        self.assertIn("later observed tool call", system_prompt)
        self.assertIn("premature loop termination", system_prompt)
        self.assertIn("orphan_tool_result", system_prompt)
        self.assertIn("rather than inventing a subsystem", system_prompt)
        self.assertLess(request.index("write_file"), request.index("run_shell"))
        self.assertIn("src/tinyagent/agent.py", request)
        self.assertEqual(json.loads(request)["benchmark_max_steps"], 12)

    def test_decision_rejects_unknown_tasks(self):
        data = {
            "layer": "context",
            "evidence": ["x"],
            "affected_tasks": ["other"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": [],
        }
        with self.assertRaisesRegex(ValueError, "current evidence"):
            EvolutionDecision.from_dict(
                data, {"task/1"}, EvolutionConfig(repo=".", run_name="test")
            )

    def test_direct_tool_removal_is_not_routed_to_architecture(self):
        architecture = {
            "layer": "architecture",
            "evidence": ["task/1 removed its output"],
            "affected_tasks": ["task/1"],
            "hypothesis": "The loop lost the output.",
            "intervention": "Change the loop.",
            "likely_files": ["src/tinyagent/agent.py"],
        }
        tools = {
            "layer": "tools",
            "evidence": ["run_shell removed the output"],
            "affected_tasks": ["task/1"],
            "hypothesis": "The shell tool permits destructive cleanup.",
            "intervention": "Clarify the shell tool contract.",
            "likely_files": [],
        }
        model = ScriptedModel(
            [
                Message("assistant", json.dumps(architecture)),
                Message("assistant", json.dumps(tools)),
            ]
        )

        decision = EvolutionDecider(model).decide(
            self._bundle("rm solution.py"),
            EvaluationReport(0.5, {}, "evaluation", "evaluation.log"),
            [],
            EvolutionConfig(repo=".", run_name="test"),
        )

        self.assertEqual(decision.layer, EvolutionLayer.TOOLS)
        self.assertIn("choose tools, not architecture", model.requests[1][-1].content)

    def test_exhausted_decision_repairs_raise_structured_error(self):
        response = {
            "layer": "architecture",
            "evidence": ["task/1 removed its output"],
            "affected_tasks": ["task/1"],
            "hypothesis": "The loop lost the output.",
            "intervention": "Change the loop.",
            "likely_files": ["src/tinyagent/agent.py"],
        }
        model = ScriptedModel(
            [Message("assistant", json.dumps(response)) for _ in range(2)]
        )

        with self.assertRaisesRegex(StructuredResponseError, "remained invalid"):
            EvolutionDecider(model).decide(
                self._bundle("rm solution.py"),
                EvaluationReport(0.5, {}, "evaluation", "evaluation.log"),
                [],
                EvolutionConfig(repo=".", run_name="test"),
            )

    def test_rejected_intervention_cannot_repeat_exactly(self):
        repeated = {
            "layer": "context",
            "evidence": ["task/1 failed"],
            "affected_tasks": ["task/1"],
            "hypothesis": "The workflow is unclear.",
            "intervention": "Clarify the workflow.",
            "likely_files": [],
        }
        revised = {**repeated, "intervention": "Require a final artifact check."}
        model = ScriptedModel(
            [
                Message("assistant", json.dumps(repeated)),
                Message("assistant", json.dumps(revised)),
            ]
        )
        history = [
            EvolutionMemoryEntry(
                1,
                "context",
                "The workflow is unclear.",
                "  CLARIFY the workflow. ",
                "rejected",
                "benchmark_rejected",
                "no gain",
                0.5,
                0.5,
            )
        ]

        decision = EvolutionDecider(model).decide(
            self._bundle(),
            EvaluationReport(0.5, {}, "evaluation", "evaluation.log"),
            history,
            EvolutionConfig(repo=".", run_name="test"),
        )

        self.assertEqual(decision.intervention, "Require a final artifact check.")
        self.assertIn("exactly repeats", model.requests[1][-1].content)

    def test_forced_architecture_bypasses_autonomous_layer_routing(self):
        response = {
            "layer": "architecture",
            "evidence": ["task/1 removed its output"],
            "affected_tasks": ["task/1"],
            "hypothesis": "The loop lost the output.",
            "intervention": "Change the loop.",
            "likely_files": ["src/tinyagent/agent.py"],
        }
        model = ScriptedModel([Message("assistant", json.dumps(response))])

        decision = EvolutionDecider(model).decide(
            self._bundle("rm solution.py"),
            EvaluationReport(0.5, {}, "evaluation", "evaluation.log"),
            [],
            EvolutionConfig(
                repo=".", run_name="test", force_layer="architecture"
            ),
        )

        self.assertEqual(decision.layer, EvolutionLayer.ARCHITECTURE)

    def test_architecture_decision_cannot_escape_mutable_source(self):
        data = {
            "layer": "architecture",
            "evidence": ["x"],
            "affected_tasks": ["task/1"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": ["eval/run.py"],
        }
        with self.assertRaisesRegex(ValueError, "mutable likely_files"):
            EvolutionDecision.from_dict(
                data, {"task/1"}, EvolutionConfig(repo=".", run_name="test")
            )

    def test_architecture_decision_requires_an_existing_source_file(self):
        data = {
            "layer": "architecture",
            "evidence": ["x"],
            "affected_tasks": ["task/1"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": ["src/tinyagent/missing.py"],
        }
        with self.assertRaisesRegex(ValueError, "existing mutable likely_files"):
            EvolutionDecision.from_dict(
                data, {"task/1"}, EvolutionConfig(repo=".", run_name="test")
            )

    def test_external_decision_ignores_source_file_suggestions(self):
        data = {
            "layer": "model",
            "evidence": ["x"],
            "affected_tasks": ["task/1"],
            "hypothesis": "x",
            "intervention": "x",
            "likely_files": ["src/tinyagent/agent.py"],
        }
        decision = EvolutionDecision.from_dict(
            data,
            {"task/1"},
            EvolutionConfig(
                repo=".", run_name="test", model_evolution=True, force_layer="model"
            ),
        )
        self.assertEqual(decision.likely_files, [])

    @staticmethod
    def _bundle(command: str = "python solution.py"):
        case = TaskEvidence(
            "task/1",
            "missing_candidate",
            False,
            "completed",
            3,
            False,
            True,
            [
                ToolEvent(1, "write_file", {"path": "solution.py"}, "wrote"),
                ToolEvent(2, "run_shell", {"command": command}, "ok"),
            ],
            "missing",
            ["artifact_missing"],
        )
        return EvidenceBundle({"passed": 0}, {"artifact_missing": 1}, [case])


if __name__ == "__main__":
    unittest.main()
