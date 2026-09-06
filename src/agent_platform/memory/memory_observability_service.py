from agent_platform.memory.contracts import (
    MemoryScope,
    MemoryType,
)
from agent_platform.memory.observability import (
    MemoryOperationTelemetryEvent,
    MemoryOperationType,
    MemoryRetrievalTelemetrySink,
    ObservabilityOutcome,
)


class MemoryObservabilityService:
    """Records safe telemetry about memory operations."""

    def __init__(
        self,
        *,
        sink: MemoryRetrievalTelemetrySink,
    ) -> None:
        self._sink = sink

    async def record_success(
        self,
        *,
        tenant_id: str,
        operation: MemoryOperationType,
        memory_type: MemoryType,
        scope: MemoryScope,
        duration_ms: float,
        record_count: int = 0,
    ) -> None:
        await self._sink.record_memory_operation(
            event=MemoryOperationTelemetryEvent(
                tenant_id=tenant_id,
                operation=operation,
                memory_type=memory_type,
                scope=scope,
                outcome=ObservabilityOutcome.SUCCESS,
                duration_ms=duration_ms,
                record_count=record_count,
            )
        )

    async def record_failure(
        self,
        *,
        tenant_id: str,
        operation: MemoryOperationType,
        memory_type: MemoryType,
        scope: MemoryScope,
        duration_ms: float,
        error: Exception,
    ) -> None:
        await self._sink.record_memory_operation(
            event=MemoryOperationTelemetryEvent(
                tenant_id=tenant_id,
                operation=operation,
                memory_type=memory_type,
                scope=scope,
                outcome=ObservabilityOutcome.FAILURE,
                duration_ms=duration_ms,
                error_type=type(error).__name__,
            )
        )
