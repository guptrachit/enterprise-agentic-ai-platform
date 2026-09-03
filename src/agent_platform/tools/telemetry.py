from dataclasses import Field, dataclass
from enum import StrEnum
from time import monotonic

from agent_platform.tools.risk import ToolRiskLevel


class ToolExecutionStatus(StrEnum):
    """Final outcome of a governed tool execution."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class ToolExecutionEvent:
    """Auditable event emitted for a governed tool execution."""

    tool_name: str
    tool_version: str
    subject_id: str
    correlation_id: str
    risk_level: ToolRiskLevel
    approval_required: bool
    approver_id: str | None
    status: ToolExecutionStatus
    duration_ms: float
    error_type: str | None = None
    attempt_count: int = 1


class ToolTelemetrySink:
    """Receives auditable tool execution events."""

    def emit(self, event: ToolExecutionEvent) -> None:
        raise NotImplementedError


class InMemoryToolTelemetrySink(ToolTelemetrySink):
    """Simple telemetry sink used for testing and local development."""

    def __init__(self) -> None:
        self._events: list[ToolExecutionEvent] = []

    def emit(self, event: ToolExecutionEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> tuple[ToolExecutionEvent, ...]:
        return tuple(self._events)


class ToolExecutionTimer:
    """Measures execution duration for telemetry."""

    def __init__(self) -> None:
        self._started_at = monotonic()

    def elapsed_ms(self) -> float:
        return (monotonic() - self._started_at) * 1000
