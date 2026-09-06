from agent_platform.memory import (
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
    KnowledgeRetrievalResult,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
)
from agent_platform.memory.retrieval_evaluation import (
    RetrievalEvaluationCase,
)
from agent_platform.memory.retrieval_evaluation_policy import (
    RetrievalEvaluationPolicy,
    RetrievalEvaluationThresholds,
)
from agent_platform.memory.retrieval_evaluation_service import (
    RetrievalEvaluationService,
)
from agent_platform.memory.retrieval_metrics import (
    RetrievalMetricCalculator,
)


def create_chunk(
    *,
    chunk_id: str,
    rank: int,
) -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        chunk=KnowledgeChunk(
            chunk_id=chunk_id,
            document_id=f"{chunk_id}-doc",
            tenant_id="tenant-001",
            content=f"content for {chunk_id}",
            chunk_index=0,
            source_type=KnowledgeSourceType.DOCUMENT,
            classification=KnowledgeClassification.INTERNAL,
        ),
        score=1.0 / rank,
        rank=rank,
    )


def create_result(
    *,
    chunk_ids: tuple[str, ...],
) -> KnowledgeRetrievalResult:
    return KnowledgeRetrievalResult(
        query=KnowledgeQuery(
            query="incident escalation",
            tenant_id="tenant-001",
            top_k=5,
        ),
        items=tuple(
            create_chunk(
                chunk_id=chunk_id,
                rank=index,
            )
            for index, chunk_id in enumerate(
                chunk_ids,
                start=1,
            )
        ),
    )


def create_service() -> RetrievalEvaluationService:
    return RetrievalEvaluationService(
        metric_calculator=RetrievalMetricCalculator(),
        policy=RetrievalEvaluationPolicy(
            thresholds=(
                RetrievalEvaluationThresholds(
                    minimum_hit_rate_at_k=1.0,
                    minimum_recall_at_k=0.5,
                    minimum_precision_at_k=0.0,
                    minimum_reciprocal_rank=0.2,
                )
            )
        ),
    )


def test_evaluation_passes_when_relevant_chunk_is_found() -> None:
    service = create_service()

    case = RetrievalEvaluationCase(
        case_id="case-001",
        query="incident escalation",
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        evaluation_k=3,
    )

    result = service.evaluate(
        case=case,
        result=create_result(
            chunk_ids=(
                "chunk-x",
                "chunk-a",
                "chunk-y",
            )
        ),
    )

    assert result.passed is True
    assert result.metrics.hit_rate_at_k == 1.0
    assert result.metrics.reciprocal_rank == 0.5


def test_evaluation_fails_when_relevant_chunk_is_missing() -> None:
    service = create_service()

    case = RetrievalEvaluationCase(
        case_id="case-002",
        query="incident escalation",
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        evaluation_k=3,
    )

    result = service.evaluate(
        case=case,
        result=create_result(
            chunk_ids=(
                "chunk-x",
                "chunk-y",
            )
        ),
    )

    assert result.passed is False
    assert result.metrics.hit_rate_at_k == 0.0


def test_evaluation_keeps_retrieved_ids_for_diagnostics() -> None:
    service = create_service()

    case = RetrievalEvaluationCase(
        case_id="case-003",
        query="incident escalation",
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
    )

    result = service.evaluate(
        case=case,
        result=create_result(
            chunk_ids=(
                "chunk-x",
                "chunk-a",
            )
        ),
    )

    assert result.retrieved_chunk_ids == (
        "chunk-x",
        "chunk-a",
    )

    assert result.relevant_retrieved_chunk_ids == ("chunk-a",)


def test_summary_aggregates_multiple_cases() -> None:
    service = create_service()

    passing_case = RetrievalEvaluationCase(
        case_id="passing",
        query="incident escalation",
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        evaluation_k=2,
    )

    failing_case = RetrievalEvaluationCase(
        case_id="failing",
        query="incident escalation",
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-z",
            }
        ),
        evaluation_k=2,
    )

    passing_result = service.evaluate(
        case=passing_case,
        result=create_result(chunk_ids=("chunk-a",)),
    )

    failing_result = service.evaluate(
        case=failing_case,
        result=create_result(chunk_ids=("chunk-x",)),
    )

    summary = service.summarize(
        results=(
            passing_result,
            failing_result,
        )
    )

    assert summary.total_cases == 2
    assert summary.passed_cases == 1
    assert summary.failed_cases == 1
    assert summary.mean_hit_rate_at_k == 0.5


def test_empty_summary_is_supported() -> None:
    service = create_service()

    summary = service.summarize(
        results=(),
    )

    assert summary.total_cases == 0
    assert summary.passed_cases == 0
    assert summary.failed_cases == 0
