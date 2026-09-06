import pytest

from agent_platform.agents.citation_validation import (
    CitationValidationService,
)
from agent_platform.agents.governed_execution import (
    GovernedAgentExecutionPort,
    GovernedAgentExecutionRequest,
    GovernedAgentExecutionResult,
)
from agent_platform.agents.governed_rag import (
    GovernedRAGAgentRequest,
)
from agent_platform.agents.governed_rag_service import (
    GovernedRAGAgentService,
)
from agent_platform.agents.retrieval import (
    AgentRetrievalRequest,
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
        assert query.tenant_id == context.tenant_id

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
                    score=0.99,
                    rank=1,
                ),
            ),
        )


class StubGovernedAgentRuntime:
    def __init__(self) -> None:
        self.last_request: GovernedAgentExecutionRequest | None = None

    async def execute(
        self,
        *,
        request: GovernedAgentExecutionRequest,
    ) -> GovernedAgentExecutionResult:
        self.last_request = request

        return GovernedAgentExecutionResult(
            response_text=("Critical incidents require immediate escalation. [E1]"),
            cited_ids=("E1",),
            tool_execution_count=0,
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


def create_retrieval_service() -> AgentRetrievalService:
    return AgentRetrievalService(
        retriever=StubKnowledgeRetriever(),
        provenance_service=(EvidenceProvenanceService()),
        context_assembly_service=(
            ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))
        ),
    )


@pytest.mark.asyncio
async def test_governed_rag_agent_end_to_end() -> None:
    runtime = StubGovernedAgentRuntime()

    assert isinstance(
        runtime,
        GovernedAgentExecutionPort,
    )

    service = GovernedRAGAgentService(
        retrieval_service=(create_retrieval_service()),
        execution_port=runtime,
        citation_validation_service=(CitationValidationService()),
    )

    result = await service.execute(
        request=GovernedRAGAgentRequest(
            user_input=("What should happen for critical incidents?"),
            retrieval_request=(
                AgentRetrievalRequest(
                    query=KnowledgeQuery(
                        query=("critical incident escalation policy"),
                        tenant_id="tenant-001",
                    ),
                    authorization_context=(create_authorization_context()),
                    context_budget=(
                        ContextBudget(
                            max_tokens=100,
                        )
                    ),
                )
            ),
        )
    )

    assert result.response_text.endswith("[E1]")

    assert result.trusted_citation_ids == ("E1",)

    assert result.citations_are_valid is True

    assert runtime.last_request is not None

    assert runtime.last_request.allowed_citation_ids == frozenset(
        {
            "E1",
        }
    )

    fragment = runtime.last_request.context_fragments[0]

    assert fragment.citation_id == "E1"

    assert fragment.source_id == "chunk-001"


class HallucinatingAgentRuntime:
    async def execute(
        self,
        *,
        request: GovernedAgentExecutionRequest,
    ) -> GovernedAgentExecutionResult:
        return GovernedAgentExecutionResult(
            response_text=("Escalate immediately. [E1] [E999]"),
            cited_ids=(
                "E1",
                "E999",
            ),
        )


@pytest.mark.asyncio
async def test_governed_rag_agent_detects_invented_citation() -> None:
    service = GovernedRAGAgentService(
        retrieval_service=(create_retrieval_service()),
        execution_port=(HallucinatingAgentRuntime()),
        citation_validation_service=(CitationValidationService()),
    )

    result = await service.execute(
        request=GovernedRAGAgentRequest(
            user_input=("What should happen for critical incidents?"),
            retrieval_request=(
                AgentRetrievalRequest(
                    query=KnowledgeQuery(
                        query=("critical incident policy"),
                        tenant_id="tenant-001",
                    ),
                    authorization_context=(create_authorization_context()),
                    context_budget=(
                        ContextBudget(
                            max_tokens=100,
                        )
                    ),
                )
            ),
        )
    )

    assert result.citations_are_valid is False

    assert result.trusted_citation_ids == ("E1",)

    assert result.citation_validation.invalid_citation_ids == ("E999",)


@pytest.mark.asyncio
async def test_rag_agent_does_not_receive_evidence_dropped_by_budget() -> None:
    runtime = StubGovernedAgentRuntime()

    service = GovernedRAGAgentService(
        retrieval_service=(create_retrieval_service()),
        execution_port=runtime,
        citation_validation_service=(CitationValidationService()),
    )

    result = await service.execute(
        request=GovernedRAGAgentRequest(
            user_input="Explain the policy.",
            retrieval_request=(
                AgentRetrievalRequest(
                    query=KnowledgeQuery(
                        query="policy",
                        tenant_id="tenant-001",
                    ),
                    authorization_context=(create_authorization_context()),
                    context_budget=(
                        ContextBudget(
                            max_tokens=1,
                        )
                    ),
                )
            ),
        )
    )

    assert result.retrieved_context.assembled_context.truncated is True

    assert runtime.last_request is not None

    assert runtime.last_request.context_fragments == ()
