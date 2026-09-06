from datetime import UTC, datetime

import pytest

from agent_platform.memory import (
    LongTermMemoryStore,
    MemoryMetadata,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RetrievalQuery,
    RetrievalResult,
)


class FakeLongTermMemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    async def save(
        self,
        *,
        record: MemoryRecord,
    ) -> MemoryRecord:
        self.records[record.memory_id] = record
        return record

    async def get(
        self,
        *,
        memory_id: str,
    ) -> MemoryRecord | None:
        return self.records.get(memory_id)

    async def search(
        self,
        *,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            items=(),
        )

    async def delete(
        self,
        *,
        memory_id: str,
    ) -> bool:
        return (
            self.records.pop(
                memory_id,
                None,
            )
            is not None
        )


def create_record() -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory-001",
        content="Preferred reporting format is PDF.",
        metadata=MemoryMetadata(
            memory_type=MemoryType.LONG_TERM,
            scope=MemoryScope.USER,
            subject_id="user-123",
            tenant_id="tenant-001",
        ),
        created_at=datetime(
            2026,
            9,
            4,
            tzinfo=UTC,
        ),
    )


def test_fake_store_satisfies_long_term_memory_protocol() -> None:
    store = FakeLongTermMemoryStore()

    assert isinstance(
        store,
        LongTermMemoryStore,
    )


@pytest.mark.asyncio
async def test_store_can_save_and_get_record() -> None:
    store = FakeLongTermMemoryStore()
    record = create_record()

    saved = await store.save(
        record=record,
    )

    loaded = await store.get(
        memory_id=record.memory_id,
    )

    assert saved == record
    assert loaded == record


@pytest.mark.asyncio
async def test_get_returns_none_for_missing_record() -> None:
    store = FakeLongTermMemoryStore()

    loaded = await store.get(
        memory_id="missing",
    )

    assert loaded is None


@pytest.mark.asyncio
async def test_delete_removes_existing_record() -> None:
    store = FakeLongTermMemoryStore()
    record = create_record()

    await store.save(
        record=record,
    )

    deleted = await store.delete(
        memory_id=record.memory_id,
    )

    loaded = await store.get(
        memory_id=record.memory_id,
    )

    assert deleted is True
    assert loaded is None


@pytest.mark.asyncio
async def test_delete_returns_false_for_missing_record() -> None:
    store = FakeLongTermMemoryStore()

    deleted = await store.delete(
        memory_id="missing",
    )

    assert deleted is False


@pytest.mark.asyncio
async def test_search_preserves_retrieval_query() -> None:
    store = FakeLongTermMemoryStore()

    query = RetrievalQuery(
        query="preferred reporting format",
        subject_id="user-123",
        tenant_id="tenant-001",
    )

    result = await store.search(
        query=query,
    )

    assert result.query == query
    assert result.items == ()
