import json
import unittest
from dataclasses import replace

from strataevo.evolution.diagnosis import EvidenceDiagnoser, EvolutionLayer
from strataevo.evolution.evidence import EvidenceBundle, TaskEvidence, ToolEvent
from strataevo.evolution.memory import EvolutionMemoryEntry, MemoryDiagnosis
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

        history = [
            EvolutionMemoryEntry(
                generation=1,
                parent_commit="parent",
                resulting_commit=None,
                decision="rejected",
                outcome_type="benchmark_rejected",
                reason="task score did not improve",
                diagnoses=[
                    MemoryDiagnosis(
                        primary_layer="tools",
                        related_layers=[],
                        problem="A previous tool hypothesis failed.",
                        affected_tasks=["HumanEval/25"],
                        proposed_direction="Change the tool description.",
                        confidence=0.7,
                    )
                ],
                plan={},
                outcome_observations=[],
                changed_paths=["src/tinyagent/workspace.py"],
                patch_path="generation-0001/changes.patch",
                patch_excerpt="- vague description\n+ precise description",
                agent_output="Changed the tool description.",
                parent_task_score=0.0,
                candidate_task_score=0.0,
            )
        ]

        report = EvidenceDiagnoser(model).diagnose(self._bundle(), history)

        diagnosis = report.diagnoses[0]
        self.assertEqual(diagnosis.primary_layer, EvolutionLayer.ARCHITECTURE)
        self.assertEqual(diagnosis.related_layers, [EvolutionLayer.TOOLS])
        self.assertEqual(report.input_tokens, 100)
        self.assertEqual(report.attempts, [report.raw_output])
        request = model.requests[0][1].content
        self.assertIn("HumanEval/25", request)
        self.assertIn("rm -f solution.py", request)
        self.assertIn('"tool_events"', request)
        self.assertIn('"result": "exit_code=0"', request)
        self.assertNotIn('"tool_sequence"', request)
        self.assertIn('"prior_evolution"', request)
        self.assertIn("task score did not improve", request)
        self.assertIn('"passed": false', request)
        self.assertIn('"requires_strict_improvement": true', request)

    def test_passed_max_steps_case_is_explicitly_an_efficiency_signal(self):
        response = {
            "diagnoses": [
                {
                    "primary_layer": "model",
                    "related_layers": [],
                    "problem": "The failed task produced an incorrect result.",
                    "evidence": ["HumanEval/25 failed its assertions."],
                    "affected_tasks": ["HumanEval/25"],
                    "proposed_direction": "Improve reasoning about the failed case.",
                    "confidence": 0.8,
                }
            ]
        }
        model = ScriptedModel([Message("assistant", json.dumps(response))])
        bundle = self._bundle()
        passed = replace(
            bundle.cases[0],
            task_id="HumanEval/26",
            status="pass",
            passed=True,
            stop_reason="max_steps",
            candidate_path="candidate.py",
            candidate_present=True,
            error="",
            signals=["max_steps"],
        )
        bundle.cases.append(passed)

        EvidenceDiagnoser(model).diagnose(bundle)

        system_prompt = model.requests[0][0].content
        request = model.requests[0][1].content
        self.assertIn("successful but potentially inefficient", system_prompt)
        self.assertIn('"task_id": "HumanEval/26"', request)
        self.assertIn('"passed": true', request)

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
        self.assertIn("response was invalid", model.requests[1][-1].content)
        self.assertNotIn("not json", model.requests[1][-1].content)

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

    def test_diagnosis_evidence_stays_within_context_budget(self):
        response = {
            "diagnoses": [
                {
                    "primary_layer": "tools",
                    "related_layers": [],
                    "problem": "The required artifact was deleted.",
                    "evidence": ["HumanEval/25 ran a destructive command."],
                    "affected_tasks": ["HumanEval/25"],
                    "proposed_direction": "Preserve required artifacts.",
                    "confidence": 0.9,
                }
            ]
        }
        model = ScriptedModel([Message("assistant", json.dumps(response))])
        bundle = self._bundle()
        bundle.cases.extend(
            replace(
                bundle.cases[0],
                task_id=f"HumanEval/{number}",
                tool_events=[
                    ToolEvent(number, f"call-{number}", "run_shell", {"command": "x" * 20_000}, "")
                ],
            )
            for number in range(26, 66)
        )
        limit = 12_000

        report = EvidenceDiagnoser(model, context_limit_chars=limit).diagnose(bundle)

        request_size = sum(len(message.content) for message in model.requests[0])
        self.assertLessEqual(request_size, limit)
        self.assertLess(report.input_case_count, len(bundle.cases))

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
            tool_events=[
                ToolEvent(1, "read", "read_file", {"path": "task.py"}, "task"),
                ToolEvent(
                    2,
                    "write",
                    "write_file",
                    {"path": "solution.py", "content": "pass\n"},
                    "Wrote 5 bytes",
                ),
                ToolEvent(
                    3,
                    "shell",
                    "run_shell",
                    {"command": "rm -f solution.py"},
                    "exit_code=0",
                ),
            ],
            error="solution.py was not created",
            generation_path="generations.jsonl",
            session_path="sessions/HumanEval_25.json",
            signals=[
                "missing_candidate",
                "artifact_missing",
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
