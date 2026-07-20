import json
import unittest

from strataevo.evolution.diagnosis import EvidenceDiagnoser, EvolutionLayer
from strataevo.evolution.evidence import EvidenceBundle, TaskEvidence
from tinyagent import Message, ModelResponse, ScriptedModel, Usage


class DiagnosisTests(unittest.TestCase):
    def test_model_diagnosis_is_parsed_into_four_layer_schema(self):
        response = {
            "diagnoses": [
                {
                    "primary_layer": "architecture",
                    "related_layers": ["tools", "architecture"],
                    "problem": "The agent reports completion after deleting its artifact.",
                    "evidence": ["HumanEval/25 removed solution.py and then completed."],
                    "affected_tasks": ["HumanEval/25"],
                    "proposed_direction": "Verify required artifacts before accepting completion.",
                    "confidence": 0.95,
                }
            ]
        }
        model = ScriptedModel(
            [
                ModelResponse(
                    Message("assistant", "```json\n" + json.dumps(response) + "\n```"),
                    Usage(100, 20),
                )
            ]
        )

        report = EvidenceDiagnoser(model).diagnose(self._bundle())

        diagnosis = report.diagnoses[0]
        self.assertEqual(diagnosis.primary_layer, EvolutionLayer.ARCHITECTURE)
        self.assertEqual(diagnosis.related_layers, [EvolutionLayer.TOOLS])
        self.assertEqual(report.input_tokens, 100)
        self.assertEqual(report.attempts, [report.raw_output])
        request = model.requests[0][1].content
        self.assertIn("HumanEval/25", request)
        self.assertIn("rm -f solution.py", request)

    def test_unknown_layer_is_rejected(self):
        response = {
            "diagnoses": [
                {
                    "primary_layer": "unknown",
                    "related_layers": [],
                    "problem": "problem",
                    "evidence": ["evidence"],
                    "affected_tasks": [],
                    "proposed_direction": "direction",
                    "confidence": 0.5,
                }
            ]
        }
        model = ScriptedModel([Message("assistant", json.dumps(response))])

        with self.assertRaises(ValueError):
            EvidenceDiagnoser(model, repair_retries=0).diagnose(self._bundle())

    def test_invalid_json_is_repaired_once(self):
        valid = {
            "diagnoses": [
                {
                    "primary_layer": "tools",
                    "related_layers": [],
                    "problem": "The required artifact was deleted.",
                    "evidence": ["HumanEval/25 ran rm -f solution.py."],
                    "affected_tasks": ["HumanEval/25"],
                    "proposed_direction": "Protect required task artifacts.",
                    "confidence": 0.9,
                }
            ]
        }
        model = ScriptedModel(
            [
                ModelResponse(Message("assistant", "not json"), Usage(10, 2)),
                ModelResponse(Message("assistant", json.dumps(valid)), Usage(20, 4)),
            ]
        )

        report = EvidenceDiagnoser(model).diagnose(self._bundle())

        self.assertEqual(report.diagnoses[0].primary_layer, EvolutionLayer.TOOLS)
        self.assertEqual(report.input_tokens, 30)
        self.assertEqual(report.output_tokens, 6)
        self.assertEqual(report.attempts[0], "not json")
        self.assertIn("JSON was invalid", model.requests[1][-1].content)

    def test_diagnosis_requires_affected_task(self):
        response = {
            "diagnoses": [
                {
                    "primary_layer": "context",
                    "related_layers": [],
                    "problem": "problem",
                    "evidence": ["evidence"],
                    "affected_tasks": [],
                    "proposed_direction": "direction",
                    "confidence": 0.5,
                }
            ]
        }
        model = ScriptedModel([Message("assistant", json.dumps(response))])

        with self.assertRaisesRegex(ValueError, "affected_tasks"):
            EvidenceDiagnoser(model, repair_retries=0).diagnose(self._bundle())

    @staticmethod
    def _bundle() -> EvidenceBundle:
        case = TaskEvidence(
            task_id="HumanEval/25",
            entry_point="factorize",
            status="missing_candidate",
            passed=False,
            stop_reason="completed",
            steps=7,
            input_tokens=100,
            output_tokens=20,
            generation_seconds=10.0,
            test_seconds=0.0,
            candidate_path=None,
            candidate_present=False,
            candidate_created=True,
            candidate_deleted=True,
            tool_sequence=["read_file", "write_file", "run_shell"],
            shell_commands=["rm -f solution.py"],
            error="solution.py was not created",
            generation_path="generations.jsonl",
            session_path="sessions/HumanEval_25.json",
            signals=[
                "missing_candidate",
                "artifact_missing",
                "artifact_deleted",
                "artifact_created_then_missing",
                "completed_without_artifact",
            ],
        )
        return EvidenceBundle(
            evaluator="humaneval",
            source_dir="evaluation",
            summary={"evaluated": 1, "passed": 0},
            signal_counts={signal: 1 for signal in case.signals},
            cases=[case],
        )


if __name__ == "__main__":
    unittest.main()
