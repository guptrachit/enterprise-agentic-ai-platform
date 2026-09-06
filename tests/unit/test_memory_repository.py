from datetime import UTC, datetime

import pytest

from agent_platform.memory import (
    InMemoryLongTermMemoryStore,
    LongTermMemoryStore,
    MemoryMetadata,
    MemoryRecord,
    MemoryRecordNotFoundError,
    MemoryRepository,
    MemoryScope,
    MemoryType,
    RetrievalQuery,
)


def create_record(
    *,
    memory_id: str = "memory-001",
    content: str = "Preferred reporting format is PDF.",
    subject_id: str = "user-123",
    tenant_id: str = "tenant-001",
    tags: frozenset[str] = frozenset(
        {
            "preference",
            "reporting",
        }
    ),
    memory_type: MemoryType = MemoryType.LONG_TERM,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        content=content,
        metadata=MemoryMetadata(
            memory_type=memory_type,
            scope=MemoryScope.USER,
            subject_id=subject_id,
            tenant_id=tenant_id,
            tags=tags,
        ),
        created_at=datetime(
            2026,
            9,
            4,
            tzinfo=UTC,
        ),
    )


def create_repository() -> MemoryRepository:
    return MemoryRepository(
        store=InMemoryLongTermMemoryStore(),
    )


def test_in_memory_store_satisfies_store_protocol() -> None:
    store = InMemoryLongTermMemoryStore()

    assert isinstance(
        store,
        LongTermMemoryStore,
    )


@pytest.mark.asyncio
async def test_repository_saves_and_loads_long_term_memory() -> None:
    repository = create_repository()
    record = create_record()

    saved = await repository.save(
        record=record,
    )

    loaded = await repository.get(
        memory_id=record.memory_id,
    )

    assert saved == record
    assert loaded == record


@pytest.mark.asyncio
async def test_repository_rejects_non_long_term_memory() -> None:
    repository = create_repository()

    record = create_record(
        memory_type=MemoryType.WORKING,
    )

    with pytest.raises(
        ValueError,
        match="only accepts LONG_TERM",
    ):
        await repository.save(
            record=record,
        )


@pytest.mark.asyncio
async def test_get_required_returns_existing_record() -> None:
    repository = create_repository()
    record = create_record()

    await repository.save(
        record=record,
    )

    loaded = await repository.get_required(
        memory_id=record.memory_id,
    )

    assert loaded == record


@pytest.mark.asyncio
async def test_get_required_raises_for_missing_record() -> None:
    repository = create_repository()

    with pytest.raises(
        MemoryRecordNotFoundError,
        match="memory-missing",
    ):
        await repository.get_required(
            memory_id="memory-missing",
        )


@pytest.mark.asyncio
async def test_repository_deletes_record() -> None:
    repository = create_repository()
    record = create_record()

    await repository.save(
        record=record,
    )

    deleted = await repository.delete(
        memory_id=record.memory_id,
    )

    loaded = await repository.get(
        memory_id=record.memory_id,
    )

    assert deleted is True
    assert loaded is None


@pytest.mark.asyncio
async def test_search_filters_by_subject_and_tenant() -> None:
    repository = create_repository()

    allowed = create_record(
        memory_id="memory-001",
        content="Preferred reporting format is PDF.",
        subject_id="user-123",
        tenant_id="tenant-001",
    )

    other_user = create_record(
        memory_id="memory-002",
        content="Preferred reporting format is PDF.",
        subject_id="user-999",
        tenant_id="tenant-001",
    )

    other_tenant = create_record(
        memory_id="memory-003",
        content="Preferred reporting format is PDF.",
        subject_id="user-123",
        tenant_id="tenant-999",
    )

    await repository.save(record=allowed)
    await repository.save(record=other_user)
    await repository.save(record=other_tenant)

    result = await repository.search(
        query=RetrievalQuery(
            query="reporting format",
            subject_id="user-123",
            tenant_id="tenant-001",
        )
    )

    assert tuple(item.record.memory_id for item in result.items) == ("memory-001",)


@pytest.mark.asyncio
async def test_search_enforces_required_tags() -> None:
    repository = create_repository()

    reporting_memory = create_record(
        memory_id="memory-001",
        content="Preferred reporting format is PDF.",
        tags=frozenset(
            {
                "preference",
                "reporting",
            }
        ),
    )

    unrelated_memory = create_record(
        memory_id="memory-002",
        content="Preferred reporting format is Excel.",
        tags=frozenset(
            {
                "preference",
            }
        ),
    )

    await repository.save(
        record=reporting_memory,
    )
    await repository.save(
        record=unrelated_memory,
    )

    result = await repository.search(
        query=RetrievalQuery(
            query="reporting format",
            subject_id="user-123",
            tenant_id="tenant-001",
            required_tags=frozenset(
                {
                    "reporting",
                }
            ),
        )
    )

    assert tuple(item.record.memory_id for item in result.items) == ("memory-001",)


@pytest.mark.asyncio
async def test_search_respects_top_k() -> None:
    repository = create_repository()

    for index in range(3):
        await repository.save(
            record=create_record(
                memory_id=f"memory-{index}",
                content=f"Reporting format preference {index}",
            )
        )

    result = await repository.search(
        query=RetrievalQuery(
            query="reporting format",
            subject_id="user-123",
            tenant_id="tenant-001",
            top_k=2,
        )
    )

    assert len(result.items) == 2

    assert tuple(item.rank for item in result.items) == (
        1,
        2,
    )
