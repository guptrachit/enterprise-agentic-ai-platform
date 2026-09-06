from agent_platform.memory.retrieval_evaluation import (
    RetrievalEvaluationMetrics,
)


class RetrievalMetricCalculator:
    """Deterministic retrieval-quality metric calculator."""

    def calculate(
        self,
        *,
        retrieved_chunk_ids: tuple[str, ...],
        expected_relevant_chunk_ids: frozenset[str],
        k: int,
    ) -> RetrievalEvaluationMetrics:
        considered = retrieved_chunk_ids[:k]

        relevant_retrieved = tuple(
            chunk_id
            for chunk_id in considered
            if chunk_id in expected_relevant_chunk_ids
        )

        hit_rate = 1.0 if relevant_retrieved else 0.0

        recall = self._recall(
            relevant_count=len(relevant_retrieved),
            expected_count=len(expected_relevant_chunk_ids),
        )

        precision = self._precision(
            relevant_count=len(relevant_retrieved),
            retrieved_count=len(considered),
        )

        reciprocal_rank = self._reciprocal_rank(
            retrieved_chunk_ids=considered,
            expected_relevant_chunk_ids=(expected_relevant_chunk_ids),
        )

        return RetrievalEvaluationMetrics(
            hit_rate_at_k=hit_rate,
            recall_at_k=recall,
            precision_at_k=precision,
            reciprocal_rank=reciprocal_rank,
        )

    @staticmethod
    def _recall(
        *,
        relevant_count: int,
        expected_count: int,
    ) -> float:
        if expected_count == 0:
            return 1.0

        return relevant_count / expected_count

    @staticmethod
    def _precision(
        *,
        relevant_count: int,
        retrieved_count: int,
    ) -> float:
        if retrieved_count == 0:
            return 0.0

        return relevant_count / retrieved_count

    @staticmethod
    def _reciprocal_rank(
        *,
        retrieved_chunk_ids: tuple[str, ...],
        expected_relevant_chunk_ids: frozenset[str],
    ) -> float:
        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1,
        ):
            if chunk_id in expected_relevant_chunk_ids:
                return 1.0 / rank

        return 0.0
