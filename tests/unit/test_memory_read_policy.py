from datetime import UTC, datetime

import pytest

from agent_platform.memory import (
    DefaultMemoryReadPolicy,
    InMemoryLongTermMemoryStore,
    MemoryMetadata,
    MemoryReadContext,
    MemoryReadDecision,
    MemoryReadDeniedError,
    MemoryReadPolicy,
    MemoryRecord,
    MemoryRepository,
    MemoryRetrievalService,
    MemoryScope,
    MemoryType,
    RetrievalQuery,
)


def create_record(
    *,
    memory_id: str = "memory-001",
    tenant_id: str | None = "tenant-001",
    subject_id: str | None = "user-123",
    scope: MemoryScope = MemoryScope.USER,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        content="Preferred reporting format is PDF.",
        metadata=MemoryMetadata(
            memory_type=MemoryType.LONG_TERM,
            scope=scope,
            subject_id=subject_id,
            tenant_id=tenant_id,
            tags=frozenset(
                {
                    "reporting",
                }
            ),
        ),
        created_at=datetime(
            2026,
            9,
            4,
            tzinfo=UTC,
        ),
    )


def test_default_read_policy_satisfies_protocol() -> None:
    policy = DefaultMemoryReadPolicy()

    assert isinstance(
        policy,
        MemoryReadPolicy,
    )


def test_query_is_allowed_for_matching_tenant_and_subject() -> None:
    policy = DefaultMemoryReadPolicy()

    evaluation = policy.authorize_query(
        query=RetrievalQuery(
            query="reporting format",
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
        context=MemoryReadContext(
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
    )

    assert evaluation.decision is MemoryReadDecision.ALLOW


def test_query_without_tenant_is_denied() -> None:
    policy = DefaultMemoryReadPolicy()

    evaluation = policy.authorize_query(
        query=RetrievalQuery(
            query="reporting format",
        ),
        context=MemoryReadContext(
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
    )

    assert evaluation.decision is MemoryReadDecision.DENY


def test_query_for_another_tenant_is_denied() -> None:
    policy = DefaultMemoryReadPolicy()

    evaluation = policy.authorize_query(
        query=RetrievalQuery(
            query="reporting format",
            tenant_id="tenant-999",
            subject_id="user-123",
        ),
        context=MemoryReadContext(
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
    )

    assert evaluation.decision is MemoryReadDecision.DENY


def test_user_scoped_record_for_another_subject_is_denied() -> None:
    policy = DefaultMemoryReadPolicy()

    evaluation = policy.authorize_record(
        record=create_record(
            subject_id="user-999",
        ),
        context=MemoryReadContext(
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
    )

    assert evaluation.decision is MemoryReadDecision.DENY


@pytest.mark.asyncio
async def test_retrieval_service_returns_authorized_memory() -> None:
    repository = MemoryRepository(
        store=InMemoryLongTermMemoryStore(),
    )

    await repository.save(
        record=create_record(),
    )

    service = MemoryRetrievalService(
        repository=repository,
        policy=DefaultMemoryReadPolicy(),
    )

    result = await service.search(
        query=RetrievalQuery(
            query="reporting format",
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
        context=MemoryReadContext(
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
    )

    assert len(result.items) == 1
    assert result.items[0].record.memory_id == "memory-001"


@pytest.mark.asyncio
async def test_retrieval_service_fails_closed_for_unauthorized_query() -> None:
    repository = MemoryRepository(
        store=InMemoryLongTermMemoryStore(),
    )

    service = MemoryRetrievalService(
        repository=repository,
        policy=DefaultMemoryReadPolicy(),
    )

    with pytest.raises(
        MemoryReadDeniedError,
        match="tenant",
    ):
        await service.search(
            query=RetrievalQuery(
                query="reporting format",
                tenant_id="tenant-999",
                subject_id="user-123",
            ),
            context=MemoryReadContext(
                tenant_id="tenant-001",
                subject_id="user-123",
            ),
        )


@pytest.mark.asyncio
async def test_retrieval_service_filters_unauthorized_returned_record() -> None:
    class LeakyStore(InMemoryLongTermMemoryStore):
        async def search(
            self,
            *,
            query: RetrievalQuery,
        ):
            from agent_platform.memory import (
                RetrievalResult,
                RetrievedItem,
            )

            leaked_record = create_record(
                memory_id="memory-leaked",
                tenant_id="tenant-999",
                subject_id="user-999",
            )

            return RetrievalResult(
                query=query,
                items=(
                    RetrievedItem(
                        record=leaked_record,
                        score=1.0,
                        rank=1,
                    ),
                ),
            )

    repository = MemoryRepository(
        store=LeakyStore(),
    )

    service = MemoryRetrievalService(
        repository=repository,
        policy=DefaultMemoryReadPolicy(),
    )

    result = await service.search(
        query=RetrievalQuery(
            query="reporting format",
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
        context=MemoryReadContext(
            tenant_id="tenant-001",
            subject_id="user-123",
        ),
    )

    assert result.items == ()
