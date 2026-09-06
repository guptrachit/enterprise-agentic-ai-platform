import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    EmbeddingVector,
    InMemoryVectorStore,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeSourceType,
    VectorRecord,
    VectorSearchQuery,
    VectorStore,
)


def create_chunk(
    *,
    chunk_id: str,
    tenant_id: str = "tenant-001",
    tags: frozenset[str] = frozenset(
        {
            "policy",
        }
    ),
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        tenant_id=tenant_id,
        content=f"Content for {chunk_id}",
        chunk_index=0,
        source_type=KnowledgeSourceType.DOCUMENT,
        classification=KnowledgeClassification.INTERNAL,
        tags=tags,
    )


def create_record(
    *,
    vector_id: str,
    vector: tuple[float, ...],
    tenant_id: str = "tenant-001",
    tags: frozenset[str] = frozenset(
        {
            "policy",
        }
    ),
) -> VectorRecord:
    return VectorRecord(
        vector_id=vector_id,
        chunk=create_chunk(
            chunk_id=f"{vector_id}-chunk",
            tenant_id=tenant_id,
            tags=tags,
        ),
        vector=EmbeddingVector(
            values=vector,
        ),
    )


def test_in_memory_vector_store_satisfies_protocol() -> None:
    store = InMemoryVectorStore()

    assert isinstance(
        store,
        VectorStore,
    )


def test_vector_search_query_requires_positive_top_k() -> None:
    with pytest.raises(ValidationError):
        VectorSearchQuery(
            vector=EmbeddingVector(
                values=(
                    1.0,
                    0.0,
                )
            ),
            tenant_id="tenant-001",
            top_k=0,
        )


@pytest.mark.asyncio
async def test_upsert_and_get_vector_record() -> None:
    store = InMemoryVectorStore()

    record = create_record(
        vector_id="vector-001",
        vector=(
            1.0,
            0.0,
        ),
    )

    saved = await store.upsert(
        record=record,
    )

    loaded = await store.get(
        vector_id="vector-001",
    )

    assert saved == record
    assert loaded == record


@pytest.mark.asyncio
async def test_search_ranks_most_similar_vector_first() -> None:
    store = InMemoryVectorStore()

    await store.upsert(
        record=create_record(
            vector_id="exact",
            vector=(
                1.0,
                0.0,
            ),
        )
    )

    await store.upsert(
        record=create_record(
            vector_id="partial",
            vector=(
                0.7,
                0.7,
            ),
        )
    )

    await store.upsert(
        record=create_record(
            vector_id="opposite",
            vector=(
                0.0,
                1.0,
            ),
        )
    )

    result = await store.search(
        query=VectorSearchQuery(
            vector=EmbeddingVector(
                values=(
                    1.0,
                    0.0,
                )
            ),
            tenant_id="tenant-001",
            top_k=3,
        )
    )

    assert tuple(item.record.vector_id for item in result.items) == (
        "exact",
        "partial",
        "opposite",
    )

    assert tuple(item.rank for item in result.items) == (
        1,
        2,
        3,
    )


@pytest.mark.asyncio
async def test_search_filters_other_tenants() -> None:
    store = InMemoryVectorStore()

    await store.upsert(
        record=create_record(
            vector_id="allowed",
            vector=(
                1.0,
                0.0,
            ),
            tenant_id="tenant-001",
        )
    )

    await store.upsert(
        record=create_record(
            vector_id="other-tenant",
            vector=(
                1.0,
                0.0,
            ),
            tenant_id="tenant-999",
        )
    )

    result = await store.search(
        query=VectorSearchQuery(
            vector=EmbeddingVector(
                values=(
                    1.0,
                    0.0,
                )
            ),
            tenant_id="tenant-001",
        )
    )

    assert tuple(item.record.vector_id for item in result.items) == ("allowed",)


@pytest.mark.asyncio
async def test_search_enforces_required_tags() -> None:
    store = InMemoryVectorStore()

    await store.upsert(
        record=create_record(
            vector_id="policy",
            vector=(
                1.0,
                0.0,
            ),
            tags=frozenset(
                {
                    "policy",
                    "support",
                }
            ),
        )
    )

    await store.upsert(
        record=create_record(
            vector_id="other",
            vector=(
                1.0,
                0.0,
            ),
            tags=frozenset(
                {
                    "support",
                }
            ),
        )
    )

    result = await store.search(
        query=VectorSearchQuery(
            vector=EmbeddingVector(
                values=(
                    1.0,
                    0.0,
                )
            ),
            tenant_id="tenant-001",
            required_tags=frozenset(
                {
                    "policy",
                }
            ),
        )
    )

    assert tuple(item.record.vector_id for item in result.items) == ("policy",)


@pytest.mark.asyncio
async def test_search_respects_top_k() -> None:
    store = InMemoryVectorStore()

    for index in range(3):
        await store.upsert(
            record=create_record(
                vector_id=f"vector-{index}",
                vector=(
                    1.0,
                    float(index) / 10,
                ),
            )
        )

    result = await store.search(
        query=VectorSearchQuery(
            vector=EmbeddingVector(
                values=(
                    1.0,
                    0.0,
                )
            ),
            tenant_id="tenant-001",
            top_k=2,
        )
    )

    assert len(result.items) == 2


@pytest.mark.asyncio
async def test_dimension_mismatch_is_rejected() -> None:
    store = InMemoryVectorStore()

    await store.upsert(
        record=create_record(
            vector_id="vector-001",
            vector=(
                1.0,
                0.0,
                0.0,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        await store.search(
            query=VectorSearchQuery(
                vector=EmbeddingVector(
                    values=(
                        1.0,
                        0.0,
                    )
                ),
                tenant_id="tenant-001",
            )
        )


@pytest.mark.asyncio
async def test_delete_removes_vector_record() -> None:
    store = InMemoryVectorStore()

    await store.upsert(
        record=create_record(
            vector_id="vector-001",
            vector=(
                1.0,
                0.0,
            ),
        )
    )

    deleted = await store.delete(
        vector_id="vector-001",
    )

    loaded = await store.get(
        vector_id="vector-001",
    )

    assert deleted is True
    assert loaded is None
