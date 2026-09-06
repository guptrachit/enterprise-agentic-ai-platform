from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.contracts import (
    MemoryRecord,
    RetrievalQuery,
)


class MemoryReadDecision(StrEnum):
    """Possible outcomes of governed memory-read evaluation."""

    ALLOW = "allow"
    DENY = "deny"


class MemoryReadContext(BaseModel):
    """Identity and tenant context used to authorize memory reads."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    subject_id: str | None = None
    tenant_id: str = Field(min_length=1)


class MemoryReadEvaluation(BaseModel):
    """Deterministic result returned by a memory-read policy."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    decision: MemoryReadDecision
    reason: str = Field(min_length=1)


@runtime_checkable
class MemoryReadPolicy(Protocol):
    """Provider-neutral policy contract for governed memory reads."""

    def authorize_query(
        self,
        *,
        query: RetrievalQuery,
        context: MemoryReadContext,
    ) -> MemoryReadEvaluation:
        """Decide whether the retrieval query may be executed."""
        ...

    def authorize_record(
        self,
        *,
        record: MemoryRecord,
        context: MemoryReadContext,
    ) -> MemoryReadEvaluation:
        """Decide whether a retrieved record may be exposed."""
        ...


class MemoryReadDeniedError(Exception):
    """Raised when memory retrieval is denied by policy."""
