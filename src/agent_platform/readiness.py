from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ReadinessStatus(StrEnum):
    """Application readiness status."""

    READY = "ready"
    NOT_READY = "not_ready"


class RuntimeHealthSnapshot(Protocol):
    """Minimal runtime health snapshot contract."""

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Return runtime health data."""


class RuntimeHealthProvider(Protocol):
    """Minimal runtime health provider contract."""

    def health_snapshot(
        self,
    ) -> RuntimeHealthSnapshot:
        """Return runtime health."""


@dataclass(frozen=True)
class ReadinessAssessment:
    """Application readiness result."""

    status: ReadinessStatus
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable readiness result."""

        return {
            "status": self.status.value,
            "reasons": list(self.reasons),
        }


def assess_runtime_readiness(
    runtime: RuntimeHealthProvider,
) -> ReadinessAssessment:
    """Assess whether the governed LLM runtime is ready."""

    snapshot = runtime.health_snapshot().to_dict()

    if not bool(
        snapshot.get(
            "resolved",
            False,
        )
    ):
        return ReadinessAssessment(
            status=ReadinessStatus.NOT_READY,
            reasons=("routing_policy_unresolved",),
        )

    return ReadinessAssessment(
        status=ReadinessStatus.READY,
        reasons=(),
    )
