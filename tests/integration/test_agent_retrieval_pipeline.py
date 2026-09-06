import pytest

from agent_platform.agents.retrieval import (
    AgentRetrievalRequest,
)
from agent_platform.agents.retrieval_service import (
    AgentRetrievalService,
)
from agent_platform.memory import (
    ContextAssemblyService,
    ContextBudget,
    DefaultKnowledgeAuthorizationPolicy,
    DeterministicEmbeddingProvider,
    DeterministicReranker,
    DeterministicTokenEstimator,
    EmbeddingService,
    EvidenceProvenanceService,
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


@pytest.mark.asyncio
async def test_agent_uses_secure_ranked_citation_aware_retrieval() -> None:
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

    await indexing_service.index_many(
        chunks=(
            KnowledgeChunk(
                chunk_id="policy",
                document_id="doc-policy",
                tenant_id="tenant-001",
                content=("Incident escalation policy requires immediate escalation."),
                chunk_index=0,
                source_type=(KnowledgeSourceType.DOCUMENT),
                classification=(KnowledgeClassification.INTERNAL),
            ),
            KnowledgeChunk(
                chunk_id="restricted",
                document_id="doc-restricted",
                tenant_id="tenant-001",
                content=("Incident escalation policy with restricted instructions."),
                chunk_index=0,
                source_type=(KnowledgeSourceType.DOCUMENT),
                classification=(KnowledgeClassification.RESTRICTED),
            ),
        )
    )

    semantic_retrieval = SemanticRetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    secure_retrieval = SecureSemanticRetrievalService(
        retrieval_service=semantic_retrieval,
        authorization_policy=(DefaultKnowledgeAuthorizationPolicy()),
    )

    ranked_retrieval = RankedKnowledgeRetrievalService(
        retrieval_service=secure_retrieval,
        ranking_service=KnowledgeRankingService(
            reranker=DeterministicReranker(),
        ),
    )

    agent_service = AgentRetrievalService(
        retriever=ranked_retrieval,
        provenance_service=(EvidenceProvenanceService()),
        context_assembly_service=(
            ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))
        ),
    )

    result = await agent_service.retrieve(
        request=AgentRetrievalRequest(
            query=KnowledgeQuery(
                query="incident escalation policy",
                tenant_id="tenant-001",
                top_k=5,
            ),
            authorization_context=(
                KnowledgeAuthorizationContext(
                    subject_id="user-123",
                    tenant_id="tenant-001",
                    allowed_classifications=frozenset(
                        {
                            KnowledgeClassification.INTERNAL,
                        }
                    ),
                    granted_scopes=frozenset(
                        {
                            "knowledge:read",
                        }
                    ),
                )
            ),
            context_budget=ContextBudget(
                max_tokens=100,
            ),
        )
    )

    retrieved_ids = tuple(item.chunk.chunk_id for item in result.retrieval_result.items)

    assert "policy" in retrieved_ids
    assert "restricted" not in retrieved_ids

    assert result.citation_registry.citation_ids == ("E1",)

    assert result.assembled_context.fragments[0].citation_id == "E1"
