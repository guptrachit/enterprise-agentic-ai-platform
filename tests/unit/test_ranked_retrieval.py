import pytest

from agent_platform.memory import (
    DefaultKnowledgeAuthorizationPolicy,
    DeterministicEmbeddingProvider,
    DeterministicReranker,
    EmbeddingService,
    InMemoryVectorStore,
    KnowledgeAuthorizationContext,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeIndexingService,
    KnowledgeQuery,
    KnowledgeRankingService,
    KnowledgeSourceType,
    RankedKnowledgeRetrievalService,
    SecureSemanticRetrievalService,
    SemanticRetrievalService,
)


def create_chunk(
    *,
    chunk_id: str,
    content: str,
    classification: KnowledgeClassification = (KnowledgeClassification.INTERNAL),
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=f"{chunk_id}-document",
        tenant_id="tenant-001",
        content=content,
        chunk_index=0,
        source_type=KnowledgeSourceType.DOCUMENT,
        classification=classification,
    )


def create_runtime():
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

    semantic_service = SemanticRetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    secure_service = SecureSemanticRetrievalService(
        retrieval_service=semantic_service,
        authorization_policy=(DefaultKnowledgeAuthorizationPolicy()),
    )

    ranked_service = RankedKnowledgeRetrievalService(
        retrieval_service=secure_service,
        ranking_service=KnowledgeRankingService(
            reranker=DeterministicReranker(),
        ),
    )

    return (
        indexing_service,
        ranked_service,
    )


def create_context() -> KnowledgeAuthorizationContext:
    return KnowledgeAuthorizationContext(
        subject_id="user-123",
        tenant_id="tenant-001",
        allowed_classifications=frozenset(
            {
                KnowledgeClassification.PUBLIC,
                KnowledgeClassification.INTERNAL,
            }
        ),
        granted_scopes=frozenset(
            {
                "knowledge:read",
            }
        ),
    )


@pytest.mark.asyncio
async def test_secure_retrieval_then_reranking() -> None:
    indexing_service, retrieval_service = create_runtime()

    await indexing_service.index_many(
        chunks=(
            create_chunk(
                chunk_id="shipment",
                content="Shipment tracking procedure.",
            ),
            create_chunk(
                chunk_id="policy",
                content=("Incident escalation policy for priority incidents."),
            ),
            create_chunk(
                chunk_id="restricted",
                content=("Incident escalation policy restricted instructions."),
                classification=(KnowledgeClassification.RESTRICTED),
            ),
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="incident escalation policy",
            tenant_id="tenant-001",
            top_k=5,
        ),
        context=create_context(),
    )

    ids = tuple(item.chunk.chunk_id for item in result.items)

    assert "restricted" not in ids
    assert ids[0] == "policy"
