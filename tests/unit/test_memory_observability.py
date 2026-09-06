import pytest

from agent_platform.memory import (
    MemoryScope,
    MemoryType,
)
from agent_platform.memory.in_memory_observability import (
    InMemoryMemoryRetrievalTelemetrySink,
)
from agent_platform.memory.memory_observability_service import (
    MemoryObservabilityService,
)
from agent_platform.memory.observability import (
    MemoryOperationType,
    MemoryRetrievalTelemetrySink,
    ObservabilityOutcome,
)


def test_in_memory_sink_satisfies_protocol() -> None:
    sink = InMemoryMemoryRetrievalTelemetrySink()

    assert isinstance(
        sink,
        MemoryRetrievalTelemetrySink,
    )


@pytest.mark.asyncio
async def test_memory_success_is_recorded() -> None:
    sink = InMemoryMemoryRetrievalTelemetrySink()

    service = MemoryObservabilityService(
        sink=sink,
    )

    await service.record_success(
        tenant_id="tenant-001",
        operation=MemoryOperationType.READ,
        memory_type=MemoryType.LONG_TERM,
        scope=MemoryScope.USER,
        duration_ms=12.5,
        record_count=3,
    )

    assert len(sink.memory_events) == 1

    event = sink.memory_events[0]

    assert event.outcome == ObservabilityOutcome.SUCCESS
    assert event.record_count == 3
    assert event.duration_ms == 12.5


@pytest.mark.asyncio
async def test_memory_failure_records_error_type_only() -> None:
    sink = InMemoryMemoryRetrievalTelemetrySink()

    service = MemoryObservabilityService(
        sink=sink,
    )

    error = ValueError("sensitive internal error details")

    await service.record_failure(
        tenant_id="tenant-001",
        operation=MemoryOperationType.WRITE,
        memory_type=MemoryType.LONG_TERM,
        scope=MemoryScope.USER,
        duration_ms=5.0,
        error=error,
    )

    event = sink.memory_events[0]

    assert event.outcome == ObservabilityOutcome.FAILURE
    assert event.error_type == "ValueError"

    serialized = event.model_dump_json()

    assert "sensitive internal error details" not in serialized
