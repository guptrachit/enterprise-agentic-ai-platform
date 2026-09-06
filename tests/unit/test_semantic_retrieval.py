import pytest

from agent_platform.memory import (
    DeterministicEmbeddingProvider,
    EmbeddingService,
    InMemoryVectorStore,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeIndexingService,
    KnowledgeQuery,
    KnowledgeSourceType,
    SemanticRetrievalService,
)


def create_chunk(
    *,
    chunk_id: str,
    content: str,
    tenant_id: str = "tenant-001",
    tags: frozenset[str] = frozenset(
        {
            "policy",
        }
    ),
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=f"{chunk_id}-document",
        tenant_id=tenant_id,
        content=content,
        chunk_index=0,
        source_type=KnowledgeSourceType.DOCUMENT,
        classification=KnowledgeClassification.INTERNAL,
        document_title=f"Document for {chunk_id}",
        tags=tags,
    )


def create_services() -> tuple[
    KnowledgeIndexingService,
    SemanticRetrievalService,
]:
    embedding_service = EmbeddingService(
        provider=DeterministicEmbeddingProvider(
            dimensions=8,
        )
    )

    vector_store = InMemoryVectorStore()

    indexing_service = KnowledgeIndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieval_service = SemanticRetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    return (
        indexing_service,
        retrieval_service,
    )


@pytest.mark.asyncio
async def test_index_chunk_creates_vector_record() -> None:
    indexing_service, _ = create_services()

    chunk = create_chunk(
        chunk_id="chunk-001",
        content="Incident escalation policy.",
    )

    record = await indexing_service.index_chunk(
        chunk=chunk,
    )

    assert record.vector_id == "chunk-001"
    assert record.chunk == chunk
    assert record.vector.dimensions == 8


@pytest.mark.asyncio
async def test_index_many_preserves_order() -> None:
    indexing_service, _ = create_services()

    chunks = (
        create_chunk(
            chunk_id="chunk-001",
            content="First policy.",
        ),
        create_chunk(
            chunk_id="chunk-002",
            content="Second policy.",
        ),
    )

    records = await indexing_service.index_many(
        chunks=chunks,
    )

    assert tuple(record.vector_id for record in records) == (
        "chunk-001",
        "chunk-002",
    )


@pytest.mark.asyncio
async def test_semantic_retrieval_returns_indexed_chunks() -> None:
    indexing_service, retrieval_service = create_services()

    chunk = create_chunk(
        chunk_id="chunk-001",
        content="Priority incidents require escalation.",
    )

    await indexing_service.index_chunk(
        chunk=chunk,
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="incident escalation",
            tenant_id="tenant-001",
        )
    )

    assert len(result.items) == 1
    assert result.items[0].chunk == chunk


@pytest.mark.asyncio
async def test_semantic_retrieval_preserves_original_query() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_chunk(
        chunk=create_chunk(
            chunk_id="chunk-001",
            content="Priority incidents require escalation.",
        )
    )

    query = KnowledgeQuery(
        query="incident escalation",
        tenant_id="tenant-001",
        top_k=3,
    )

    result = await retrieval_service.retrieve(
        query=query,
    )

    assert result.query == query


@pytest.mark.asyncio
async def test_semantic_retrieval_preserves_rank_and_score() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_many(
        chunks=(
            create_chunk(
                chunk_id="chunk-001",
                content="Incident escalation policy.",
            ),
            create_chunk(
                chunk_id="chunk-002",
                content="Shipment tracking procedure.",
            ),
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="incident escalation policy",
            tenant_id="tenant-001",
            top_k=2,
        )
    )

    assert len(result.items) == 2
    assert result.items[0].rank == 1
    assert result.items[1].rank == 2
    assert result.items[0].score is not None


@pytest.mark.asyncio
async def test_semantic_retrieval_respects_tenant_boundary() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_many(
        chunks=(
            create_chunk(
                chunk_id="tenant-a",
                content="Escalation policy.",
                tenant_id="tenant-001",
            ),
            create_chunk(
                chunk_id="tenant-b",
                content="Escalation policy.",
                tenant_id="tenant-999",
            ),
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="escalation policy",
            tenant_id="tenant-001",
        )
    )

    assert tuple(item.chunk.chunk_id for item in result.items) == ("tenant-a",)


@pytest.mark.asyncio
async def test_semantic_retrieval_respects_required_tags() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_many(
        chunks=(
            create_chunk(
                chunk_id="policy",
                content="Escalation policy.",
                tags=frozenset(
                    {
                        "policy",
                        "support",
                    }
                ),
            ),
            create_chunk(
                chunk_id="manual",
                content="Escalation manual.",
                tags=frozenset(
                    {
                        "manual",
                    }
                ),
            ),
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="escalation",
            tenant_id="tenant-001",
            required_tags=frozenset(
                {
                    "policy",
                }
            ),
        )
    )

    assert tuple(item.chunk.chunk_id for item in result.items) == ("policy",)


@pytest.mark.asyncio
async def test_semantic_retrieval_respects_top_k() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_many(
        chunks=tuple(
            create_chunk(
                chunk_id=f"chunk-{index}",
                content=f"Policy content {index}",
            )
            for index in range(5)
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="policy content",
            tenant_id="tenant-001",
            top_k=2,
        )
    )

    assert len(result.items) == 2
