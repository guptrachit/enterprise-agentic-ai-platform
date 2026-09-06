import pytest

from agent_platform.memory import (
    DefaultKnowledgeAuthorizationPolicy,
    DeterministicEmbeddingProvider,
    EmbeddingService,
    InMemoryVectorStore,
    KnowledgeAccessDeniedError,
    KnowledgeAuthorizationContext,
    KnowledgeAuthorizationPolicy,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeIndexingService,
    KnowledgeQuery,
    KnowledgeSourceType,
    SecureSemanticRetrievalService,
    SemanticRetrievalService,
)


def create_chunk(
    *,
    chunk_id: str,
    tenant_id: str = "tenant-001",
    classification: KnowledgeClassification = (KnowledgeClassification.INTERNAL),
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=f"{chunk_id}-document",
        tenant_id=tenant_id,
        content="Incident escalation policy.",
        chunk_index=0,
        source_type=KnowledgeSourceType.DOCUMENT,
        classification=classification,
        tags=frozenset(
            {
                "policy",
            }
        ),
    )


def create_services():
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

    semantic_retrieval = SemanticRetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    secure_retrieval = SecureSemanticRetrievalService(
        retrieval_service=semantic_retrieval,
        authorization_policy=(DefaultKnowledgeAuthorizationPolicy()),
    )

    return (
        indexing_service,
        secure_retrieval,
    )


def create_context(
    *,
    tenant_id: str = "tenant-001",
    classifications: frozenset[KnowledgeClassification] = frozenset(
        {
            KnowledgeClassification.PUBLIC,
            KnowledgeClassification.INTERNAL,
        }
    ),
    scopes: frozenset[str] = frozenset(
        {
            "knowledge:read",
        }
    ),
) -> KnowledgeAuthorizationContext:
    return KnowledgeAuthorizationContext(
        subject_id="user-123",
        tenant_id=tenant_id,
        allowed_classifications=classifications,
        granted_scopes=scopes,
    )


def test_default_policy_satisfies_protocol() -> None:
    policy = DefaultKnowledgeAuthorizationPolicy()

    assert isinstance(
        policy,
        KnowledgeAuthorizationPolicy,
    )


@pytest.mark.asyncio
async def test_authorized_retrieval_returns_allowed_chunk() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_chunk(
        chunk=create_chunk(
            chunk_id="internal-policy",
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="incident escalation",
            tenant_id="tenant-001",
        ),
        context=create_context(),
    )

    assert len(result.items) == 1
    assert result.items[0].chunk.chunk_id == "internal-policy"


@pytest.mark.asyncio
async def test_missing_knowledge_read_scope_is_denied() -> None:
    _, retrieval_service = create_services()

    with pytest.raises(
        KnowledgeAccessDeniedError,
        match="knowledge:read",
    ):
        await retrieval_service.retrieve(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-001",
            ),
            context=create_context(
                scopes=frozenset(),
            ),
        )


@pytest.mark.asyncio
async def test_cross_tenant_query_is_denied() -> None:
    _, retrieval_service = create_services()

    with pytest.raises(
        KnowledgeAccessDeniedError,
        match="another tenant",
    ):
        await retrieval_service.retrieve(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-999",
            ),
            context=create_context(
                tenant_id="tenant-001",
            ),
        )


@pytest.mark.asyncio
async def test_restricted_chunk_is_not_exposed_to_internal_user() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_many(
        chunks=(
            create_chunk(
                chunk_id="internal",
                classification=(KnowledgeClassification.INTERNAL),
            ),
            create_chunk(
                chunk_id="restricted",
                classification=(KnowledgeClassification.RESTRICTED),
            ),
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="incident escalation",
            tenant_id="tenant-001",
        ),
        context=create_context(),
    )

    assert tuple(item.chunk.chunk_id for item in result.items) == ("internal",)


@pytest.mark.asyncio
async def test_restricted_user_can_retrieve_restricted_chunk() -> None:
    indexing_service, retrieval_service = create_services()

    await indexing_service.index_chunk(
        chunk=create_chunk(
            chunk_id="restricted",
            classification=(KnowledgeClassification.RESTRICTED),
        )
    )

    result = await retrieval_service.retrieve(
        query=KnowledgeQuery(
            query="incident escalation",
            tenant_id="tenant-001",
        ),
        context=create_context(
            classifications=frozenset(
                {
                    KnowledgeClassification.INTERNAL,
                    KnowledgeClassification.RESTRICTED,
                }
            ),
        ),
    )

    assert len(result.items) == 1
    assert result.items[0].chunk.chunk_id == "restricted"


@pytest.mark.asyncio
async def test_query_cannot_request_more_classification_access() -> None:
    _, retrieval_service = create_services()

    with pytest.raises(
        KnowledgeAccessDeniedError,
        match="outside caller authorization",
    ):
        await retrieval_service.retrieve(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-001",
                allowed_classifications=frozenset(
                    {
                        KnowledgeClassification.RESTRICTED,
                    }
                ),
            ),
            context=create_context(),
        )
