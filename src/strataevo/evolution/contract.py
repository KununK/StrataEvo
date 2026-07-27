"""Describe the objective of one evolution benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    """The benchmark name and task objective shown to the evolution agent."""

    benchmark: str
    objective: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
