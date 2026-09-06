from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.contracts import MemoryRecord


class MemoryOperation(StrEnum):
    """Governed operations that may be performed on memory."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"


class MemoryAuthorizationContext(BaseModel):
    """Identity boundary used for governed memory access."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    subject_id: str | None = None
    tenant_id: str = Field(min_length=1)
    granted_scopes: frozenset[str] = frozenset()


class MemoryAuthorizationDecision(StrEnum):
    """Possible authorization outcomes."""

    ALLOW = "allow"
    DENY = "deny"


class MemoryAuthorizationEvaluation(BaseModel):
    """Result returned by memory authorization policy."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    decision: MemoryAuthorizationDecision
    reason: str = Field(min_length=1)


@runtime_checkable
class MemoryAuthorizationPolicy(Protocol):
    """Provider-neutral authorization contract for memory operations."""

    def authorize_record(
        self,
        *,
        operation: MemoryOperation,
        record: MemoryRecord,
        context: MemoryAuthorizationContext,
    ) -> MemoryAuthorizationEvaluation:
        """Authorize an operation against a concrete memory record."""
        ...


class MemoryAuthorizationDeniedError(Exception):
    """Raised when a memory operation is denied."""
