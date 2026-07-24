"""Declare which repository changes a benchmark can observe."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ChangeImpact:
    """Classify a patch by when the active benchmark can observe it."""

    direct_paths: tuple[str, ...]
    deferred_paths: tuple[str, ...]
    unclassified_paths: tuple[str, ...]

    @property
    def is_direct_only(self) -> bool:
        return bool(self.direct_paths) and not self.deferred_paths and not self.unclassified_paths

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "direct_paths": list(self.direct_paths),
            "deferred_paths": list(self.deferred_paths),
            "unclassified_paths": list(self.unclassified_paths),
            "is_direct_only": self.is_direct_only,
        }


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    """The code boundary and objective of one benchmark."""

    benchmark: str
    objective: str
    direct_paths: tuple[str, ...]
    deferred_paths: tuple[str, ...] = ()

    def classify(self, changed_paths: list[str]) -> ChangeImpact:
        direct: list[str] = []
        deferred: list[str] = []
        unclassified: list[str] = []
        for path in changed_paths:
            if _matches_any(path, self.direct_paths):
                direct.append(path)
            elif _matches_any(path, self.deferred_paths):
                deferred.append(path)
            else:
                unclassified.append(path)
        return ChangeImpact(tuple(direct), tuple(deferred), tuple(unclassified))

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "direct_paths": list(self.direct_paths),
            "deferred_paths": list(self.deferred_paths),
        }


def _matches_any(path: str, roots: tuple[str, ...]) -> bool:
    normalized = path.strip("/")
    return any(
        normalized == root.strip("/") or normalized.startswith(root.strip("/") + "/")
        for root in roots
    )
