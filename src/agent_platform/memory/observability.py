from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.contracts import (
    MemoryScope,
    MemoryType,
)


class ObservabilityOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class MemoryOperationType(StrEnum):
    READ = "read"
    SEARCH = "search"
    WRITE = "write"
    DELETE = "delete"


class RetrievalTelemetryEvent(BaseModel):
    """Safe operational telemetry for one retrieval operation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tenant_id: str = Field(min_length=1)

    operation: str = "knowledge_retrieval"

    outcome: ObservabilityOutcome

    duration_ms: float = Field(ge=0.0)

    retrieved_count: int = Field(ge=0)

    context_fragment_count: int = Field(ge=0)

    budget_dropped_count: int = Field(ge=0)

    used_tokens: int = Field(ge=0)

    max_tokens: int = Field(ge=1)

    truncated: bool

    error_type: str | None = None


class MemoryOperationTelemetryEvent(BaseModel):
    """Safe operational telemetry for one memory operation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tenant_id: str = Field(min_length=1)

    operation: MemoryOperationType

    memory_type: MemoryType

    scope: MemoryScope

    outcome: ObservabilityOutcome

    duration_ms: float = Field(ge=0.0)

    record_count: int = Field(
        default=0,
        ge=0,
    )

    error_type: str | None = None


@runtime_checkable
class MemoryRetrievalTelemetrySink(Protocol):
    """Provider-neutral telemetry destination."""

    async def record_retrieval(
        self,
        *,
        event: RetrievalTelemetryEvent,
    ) -> None: ...

    async def record_memory_operation(
        self,
        *,
        event: MemoryOperationTelemetryEvent,
    ) -> None: ...
