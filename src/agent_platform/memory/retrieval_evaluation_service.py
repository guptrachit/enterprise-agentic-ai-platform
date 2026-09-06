from agent_platform.memory.knowledge import (
    KnowledgeRetrievalResult,
)
from agent_platform.memory.retrieval_evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
    RetrievalEvaluationSummary,
)
from agent_platform.memory.retrieval_evaluation_policy import (
    RetrievalEvaluationPolicy,
)
from agent_platform.memory.retrieval_metrics import (
    RetrievalMetricCalculator,
)


class RetrievalEvaluationService:
    """Evaluates retrieval results against known relevance labels."""

    def __init__(
        self,
        *,
        metric_calculator: RetrievalMetricCalculator,
        policy: RetrievalEvaluationPolicy,
    ) -> None:
        self._metric_calculator = metric_calculator
        self._policy = policy

    def evaluate(
        self,
        *,
        case: RetrievalEvaluationCase,
        result: KnowledgeRetrievalResult,
    ) -> RetrievalEvaluationResult:
        retrieved_chunk_ids = tuple(item.chunk.chunk_id for item in result.items)

        relevant_retrieved_chunk_ids = tuple(
            chunk_id
            for chunk_id in retrieved_chunk_ids[: case.evaluation_k]
            if chunk_id in case.expected_relevant_chunk_ids
        )

        metrics = self._metric_calculator.calculate(
            retrieved_chunk_ids=retrieved_chunk_ids,
            expected_relevant_chunk_ids=(case.expected_relevant_chunk_ids),
            k=case.evaluation_k,
        )

        return RetrievalEvaluationResult(
            case=case,
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_retrieved_chunk_ids=(relevant_retrieved_chunk_ids),
            metrics=metrics,
            passed=self._policy.passes(
                metrics=metrics,
            ),
        )

    def summarize(
        self,
        *,
        results: tuple[RetrievalEvaluationResult, ...],
    ) -> RetrievalEvaluationSummary:
        total = len(results)

        if total == 0:
            return RetrievalEvaluationSummary(
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                mean_hit_rate_at_k=0.0,
                mean_recall_at_k=0.0,
                mean_precision_at_k=0.0,
                mean_reciprocal_rank=0.0,
            )

        passed = sum(1 for result in results if result.passed)

        return RetrievalEvaluationSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            mean_hit_rate_at_k=(
                sum(result.metrics.hit_rate_at_k for result in results) / total
            ),
            mean_recall_at_k=(
                sum(result.metrics.recall_at_k for result in results) / total
            ),
            mean_precision_at_k=(
                sum(result.metrics.precision_at_k for result in results) / total
            ),
            mean_reciprocal_rank=(
                sum(result.metrics.reciprocal_rank for result in results) / total
            ),
        )
