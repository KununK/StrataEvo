"""Describe a benchmark and how repository changes affect it."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ChangeEffect:
    scope: str
    layers: tuple[str, ...]
    activation: str
    benchmark_effect: str


DEFAULT_CHANGE_EFFECTS = (
    ChangeEffect(
        scope="src/tinyagent/**",
        layers=("context", "tools", "architecture"),
        activation="immediate: benchmark tasks start a fresh Python process",
        benchmark_effect="directly changes the task-solving Agent",
    ),
    ChangeEffect(
        scope="src/strataevo/evolution/**",
        layers=("context", "tools", "architecture"),
        activation="next generation: the current evolution worker already imported this code",
        benchmark_effect="changes the evolution controller, not the current task-solving Agent",
    ),
    ChangeEffect(
        scope="eval/**",
        layers=("architecture",),
        activation="immediate: candidate evaluation starts a fresh benchmark process",
        benchmark_effect=(
            "changes measurement; a higher score caused by evaluator changes is not evidence "
            "that the Agent improved"
        ),
    ),
    ChangeEffect(
        scope="tests/**",
        layers=("architecture",),
        activation="immediate for validation only",
        benchmark_effect="does not change task-solving behavior",
    ),
    ChangeEffect(
        scope="new or otherwise unreferenced files",
        layers=(),
        activation="none until active runtime code imports or reads the file",
        benchmark_effect="standalone answers for benchmark tasks are not loaded automatically",
    ),
    ChangeEffect(
        scope="model weights or adapters",
        layers=("model",),
        activation="unavailable through a source edit unless an explicit TTA load path exists",
        benchmark_effect="repository edits alone do not change the served model weights",
    ),
)


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    """The benchmark name and task objective shown to the evolution agent."""

    benchmark: str
    objective: str
    change_effects: tuple[ChangeEffect, ...] = DEFAULT_CHANGE_EFFECTS

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
