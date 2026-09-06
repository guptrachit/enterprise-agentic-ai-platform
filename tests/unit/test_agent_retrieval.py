import pytest

from agent_platform.agents.retrieval import (
    AgentKnowledgeRetriever,
    AgentRetrievalRequest,
)
from agent_platform.agents.retrieval_context import (
    AgentRetrievalExecutionContext,
)
from agent_platform.agents.retrieval_service import (
    AgentRetrievalService,
)
from agent_platform.memory import (
    ContextAssemblyService,
    ContextBudget,
    DeterministicTokenEstimator,
    EvidenceProvenanceService,
    KnowledgeAuthorizationContext,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
    KnowledgeRetrievalResult,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
)


class StubKnowledgeRetriever:
    async def retrieve(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeRetrievalResult:
        assert context.tenant_id == query.tenant_id

        return KnowledgeRetrievalResult(
            query=query,
            items=(
                RetrievedKnowledgeChunk(
                    chunk=KnowledgeChunk(
                        chunk_id="chunk-001",
                        document_id="doc-001",
                        tenant_id=query.tenant_id,
                        content=("Critical incidents require immediate escalation."),
                        chunk_index=0,
                        source_type=(KnowledgeSourceType.DOCUMENT),
                        classification=(KnowledgeClassification.INTERNAL),
                    ),
                    score=0.95,
                    rank=1,
                ),
            ),
        )


def create_authorization_context() -> KnowledgeAuthorizationContext:
    return KnowledgeAuthorizationContext(
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


def create_service() -> AgentRetrievalService:
    return AgentRetrievalService(
        retriever=StubKnowledgeRetriever(),
        provenance_service=(EvidenceProvenanceService()),
        context_assembly_service=(
            ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))
        ),
    )


def test_stub_retriever_satisfies_protocol() -> None:
    assert isinstance(
        StubKnowledgeRetriever(),
        AgentKnowledgeRetriever,
    )


@pytest.mark.asyncio
async def test_agent_retrieval_builds_governed_context() -> None:
    service = create_service()

    result = await service.retrieve(
        request=AgentRetrievalRequest(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-001",
            ),
            authorization_context=(create_authorization_context()),
            context_budget=ContextBudget(
                max_tokens=100,
            ),
        )
    )

    assert len(result.retrieval_result.items) == 1

    assert result.retrieval_result.items[0].chunk.chunk_id == "chunk-001"


@pytest.mark.asyncio
async def test_agent_retrieval_creates_citation_registry() -> None:
    service = create_service()

    result = await service.retrieve(
        request=AgentRetrievalRequest(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-001",
            ),
            authorization_context=(create_authorization_context()),
            context_budget=ContextBudget(
                max_tokens=100,
            ),
        )
    )

    assert result.citation_registry.citation_ids == ("E1",)


@pytest.mark.asyncio
async def test_agent_context_fragment_keeps_citation() -> None:
    service = create_service()

    result = await service.retrieve(
        request=AgentRetrievalRequest(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-001",
            ),
            authorization_context=(create_authorization_context()),
            context_budget=ContextBudget(
                max_tokens=100,
            ),
        )
    )

    assert result.assembled_context.fragments[0].citation_id == "E1"


@pytest.mark.asyncio
async def test_agent_context_respects_token_budget() -> None:
    service = create_service()

    result = await service.retrieve(
        request=AgentRetrievalRequest(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-001",
            ),
            authorization_context=(create_authorization_context()),
            context_budget=ContextBudget(
                max_tokens=2,
            ),
        )
    )

    assert result.assembled_context.used_tokens <= 2

    assert result.assembled_context.truncated is True


@pytest.mark.asyncio
async def test_agent_retrieval_execution_context_exposes_citations() -> None:
    service = create_service()

    retrieved = await service.retrieve(
        request=AgentRetrievalRequest(
            query=KnowledgeQuery(
                query="incident escalation",
                tenant_id="tenant-001",
            ),
            authorization_context=(create_authorization_context()),
            context_budget=ContextBudget(
                max_tokens=100,
            ),
        )
    )

    execution_context = AgentRetrievalExecutionContext(
        retrieved_context=retrieved,
    )

    assert execution_context.has_retrieval_context is True

    assert execution_context.citation_ids == ("E1",)


def test_empty_agent_retrieval_context() -> None:
    context = AgentRetrievalExecutionContext()

    assert context.has_retrieval_context is False
    assert context.citation_ids == ()
