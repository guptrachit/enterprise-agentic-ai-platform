from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolAuditDecision(StrEnum):
    """Governance decision produced before tool invocation."""

    AUTHORIZED = "authorized"
    AUTHORIZATION_DENIED = "authorization_denied"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    REJECTED = "rejected"


class ToolAuditEvent(BaseModel):
    """Immutable audit record for a governed tool decision."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)

    correlation_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)

    decision: ToolAuditDecision

    approver_id: str | None = None


class ToolAuditSink(Protocol):
    """Destination for governed tool audit events."""

    def emit(
        self,
        event: ToolAuditEvent,
    ) -> None:
        """Persist or export an audit event."""


class InMemoryToolAuditSink:
    """In-memory audit sink used by tests."""

    def __init__(self) -> None:
        self._events: list[ToolAuditEvent] = []

    @property
    def events(self) -> tuple[ToolAuditEvent, ...]:
        return tuple(self._events)

    def emit(
        self,
        event: ToolAuditEvent,
    ) -> None:
        self._events.append(event)
