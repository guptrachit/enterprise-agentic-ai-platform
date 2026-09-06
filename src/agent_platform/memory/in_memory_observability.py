from agent_platform.memory.observability import (
    MemoryOperationTelemetryEvent,
    RetrievalTelemetryEvent,
)


class InMemoryMemoryRetrievalTelemetrySink:
    """Deterministic telemetry sink for tests and local development."""

    def __init__(self) -> None:
        self._retrieval_events: list[RetrievalTelemetryEvent] = []

        self._memory_events: list[MemoryOperationTelemetryEvent] = []

    @property
    def retrieval_events(
        self,
    ) -> tuple[RetrievalTelemetryEvent, ...]:
        return tuple(self._retrieval_events)

    @property
    def memory_events(
        self,
    ) -> tuple[MemoryOperationTelemetryEvent, ...]:
        return tuple(self._memory_events)

    async def record_retrieval(
        self,
        *,
        event: RetrievalTelemetryEvent,
    ) -> None:
        self._retrieval_events.append(event)

    async def record_memory_operation(
        self,
        *,
        event: MemoryOperationTelemetryEvent,
    ) -> None:
        self._memory_events.append(event)
