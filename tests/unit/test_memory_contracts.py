from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    ContextItem,
    MemoryMetadata,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
)


def create_record() -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory-001",
        content="The preferred reporting format is PDF.",
        metadata=MemoryMetadata(
            memory_type=MemoryType.LONG_TERM,
            scope=MemoryScope.USER,
            subject_id="user-123",
            tenant_id="tenant-001",
            tags=frozenset(
                {
                    "preference",
                    "reporting",
                }
            ),
        ),
        created_at=datetime(
            2026,
            9,
            3,
            tzinfo=UTC,
        ),
    )


def test_context_item_is_immutable() -> None:
    item = ContextItem(
        content="Current customer is CUST-123.",
        source="working-memory",
    )

    with pytest.raises(ValidationError):
        item.content = "Changed"


def test_memory_record_contains_governance_metadata() -> None:
    record = create_record()

    assert record.metadata.memory_type is MemoryType.LONG_TERM
    assert record.metadata.scope is MemoryScope.USER
    assert record.metadata.subject_id == "user-123"
    assert record.metadata.tenant_id == "tenant-001"


def test_retrieval_query_requires_positive_top_k() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(
            query="report format",
            top_k=0,
        )


def test_retrieved_item_requires_positive_rank() -> None:
    with pytest.raises(ValidationError):
        RetrievedItem(
            record=create_record(),
            score=0.95,
            rank=0,
        )


def test_retrieval_result_preserves_query_and_order() -> None:
    query = RetrievalQuery(
        query="preferred report format",
        top_k=3,
        subject_id="user-123",
        tenant_id="tenant-001",
    )

    item = RetrievedItem(
        record=create_record(),
        score=0.97,
        rank=1,
    )

    result = RetrievalResult(
        query=query,
        items=(item,),
    )

    assert result.query == query
    assert result.items == (item,)
    assert result.items[0].rank == 1
