import pytest

from agent_platform.agents.observable_retrieval_service import (
    ObservableAgentRetrievalService,
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
from agent_platform.memory.in_memory_observability import (
    InMemoryMemoryRetrievalTelemetrySink,
)
from agent_platform.memory.observability import (
    ObservabilityOutcome,
)


class StubRetriever:
    async def retrieve(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeRetrievalResult:
        return KnowledgeRetrievalResult(
            query=query,
            items=(
                RetrievedKnowledgeChunk(
                    chunk=KnowledgeChunk(
                        chunk_id="chunk-001",
                        document_id="doc-001",
                        tenant_id=query.tenant_id,
                        content=("Escalation policy information."),
                        chunk_index=0,
                        source_type=(KnowledgeSourceType.DOCUMENT),
                        classification=(KnowledgeClassification.INTERNAL),
                    ),
                    score=0.95,
                    rank=1,
                ),
            ),
        )


def create_request(
    *,
    max_tokens: int = 100,
) -> AgentRetrievalRequest:
    return AgentRetrievalRequest(
        query=KnowledgeQuery(
            query="incident escalation",
            tenant_id="tenant-001",
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
            max_tokens=max_tokens,
        ),
    )


def create_service(
    *,
    sink: InMemoryMemoryRetrievalTelemetrySink,
) -> ObservableAgentRetrievalService:
    base_service = AgentRetrievalService(
        retriever=StubRetriever(),
        provenance_service=(EvidenceProvenanceService()),
        context_assembly_service=(
            ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))
        ),
    )

    return ObservableAgentRetrievalService(
        service=base_service,
        telemetry_sink=sink,
    )


@pytest.mark.asyncio
async def test_successful_retrieval_records_metrics() -> None:
    sink = InMemoryMemoryRetrievalTelemetrySink()

    service = create_service(
        sink=sink,
    )

    result = await service.retrieve(
        request=create_request(),
    )

    assert len(result.retrieval_result.items) == 1
    assert len(sink.retrieval_events) == 1

    event = sink.retrieval_events[0]

    assert event.outcome == ObservabilityOutcome.SUCCESS
    assert event.retrieved_count == 1
    assert event.context_fragment_count == 1
    assert event.budget_dropped_count == 0
    assert event.used_tokens > 0
    assert event.duration_ms >= 0


@pytest.mark.asyncio
async def test_token_budget_truncation_is_observable() -> None:
    sink = InMemoryMemoryRetrievalTelemetrySink()

    service = create_service(
        sink=sink,
    )

    await service.retrieve(
        request=create_request(
            max_tokens=1,
        ),
    )

    event = sink.retrieval_events[0]

    assert event.truncated is True
    assert event.context_fragment_count == 0
    assert event.budget_dropped_count == 1


@pytest.mark.asyncio
async def test_query_text_is_not_written_to_telemetry() -> None:
    sink = InMemoryMemoryRetrievalTelemetrySink()

    service = create_service(
        sink=sink,
    )

    await service.retrieve(
        request=create_request(),
    )

    serialized = sink.retrieval_events[0].model_dump_json()

    assert "incident escalation" not in serialized
    assert "Escalation policy information" not in serialized


class FailingRetriever:
    async def retrieve(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeRetrievalResult:
        raise RuntimeError("secret backend failure details")


@pytest.mark.asyncio
async def test_retrieval_failure_is_recorded_without_error_message() -> None:
    sink = InMemoryMemoryRetrievalTelemetrySink()

    base_service = AgentRetrievalService(
        retriever=FailingRetriever(),
        provenance_service=(EvidenceProvenanceService()),
        context_assembly_service=(
            ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))
        ),
    )

    service = ObservableAgentRetrievalService(
        service=base_service,
        telemetry_sink=sink,
    )

    with pytest.raises(
        RuntimeError,
        match="secret backend failure details",
    ):
        await service.retrieve(
            request=create_request(),
        )

    assert len(sink.retrieval_events) == 1

    event = sink.retrieval_events[0]

    assert event.outcome == ObservabilityOutcome.FAILURE
    assert event.error_type == "RuntimeError"

    serialized = event.model_dump_json()

    assert "secret backend failure details" not in serialized
