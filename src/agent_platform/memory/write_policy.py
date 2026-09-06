from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.contracts import (
    MemoryMetadata,
    MemoryRecord,
)


class MemoryWriteDecision(StrEnum):
    """Possible outcomes of governed memory-write evaluation."""

    ALLOW = "allow"
    DENY = "deny"


class MemoryWriteRequest(BaseModel):
    """Candidate memory proposed for durable persistence."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    content: str = Field(min_length=1)
    metadata: MemoryMetadata
    source: str = Field(min_length=1)


class MemoryWriteEvaluation(BaseModel):
    """Deterministic result returned by a memory-write policy."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    decision: MemoryWriteDecision
    reason: str = Field(min_length=1)


@runtime_checkable
class MemoryWritePolicy(Protocol):
    """Provider-neutral policy contract for durable memory writes."""

    def evaluate(
        self,
        *,
        request: MemoryWriteRequest,
    ) -> MemoryWriteEvaluation:
        """Decide whether a candidate memory may be persisted."""
        ...


class MemoryWriteDeniedError(Exception):
    """Raised when durable memory persistence is denied by policy."""


class MemoryWriteService:
    """Governed entry point for converting candidates into durable memory."""

    def __init__(
        self,
        *,
        policy: MemoryWritePolicy,
    ) -> None:
        self._policy = policy

    def authorize(
        self,
        *,
        request: MemoryWriteRequest,
    ) -> MemoryWriteEvaluation:
        """Evaluate a candidate memory and fail closed when denied."""

        evaluation = self._policy.evaluate(
            request=request,
        )

        if evaluation.decision is MemoryWriteDecision.DENY:
            raise MemoryWriteDeniedError(evaluation.reason)

        return evaluation

    def build_record(
        self,
        *,
        memory_id: str,
        created_at,
        request: MemoryWriteRequest,
    ) -> MemoryRecord:
        """Create a durable record only after write authorization."""

        self.authorize(
            request=request,
        )

        return MemoryRecord(
            memory_id=memory_id,
            content=request.content,
            metadata=request.metadata,
            created_at=created_at,
            attributes={
                "write_source": request.source,
            },
        )
