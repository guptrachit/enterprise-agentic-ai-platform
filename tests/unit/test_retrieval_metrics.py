import pytest

from agent_platform.memory.retrieval_metrics import (
    RetrievalMetricCalculator,
)


def test_hit_rate_is_one_when_relevant_item_is_retrieved() -> None:
    calculator = RetrievalMetricCalculator()

    metrics = calculator.calculate(
        retrieved_chunk_ids=(
            "chunk-x",
            "chunk-a",
        ),
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        k=2,
    )

    assert metrics.hit_rate_at_k == 1.0


def test_hit_rate_is_zero_when_no_relevant_item_is_retrieved() -> None:
    calculator = RetrievalMetricCalculator()

    metrics = calculator.calculate(
        retrieved_chunk_ids=(
            "chunk-x",
            "chunk-y",
        ),
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        k=2,
    )

    assert metrics.hit_rate_at_k == 0.0


def test_recall_at_k() -> None:
    calculator = RetrievalMetricCalculator()

    metrics = calculator.calculate(
        retrieved_chunk_ids=(
            "chunk-a",
            "chunk-x",
        ),
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
                "chunk-b",
            }
        ),
        k=2,
    )

    assert metrics.recall_at_k == 0.5


def test_precision_at_k() -> None:
    calculator = RetrievalMetricCalculator()

    metrics = calculator.calculate(
        retrieved_chunk_ids=(
            "chunk-a",
            "chunk-x",
            "chunk-y",
        ),
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        k=3,
    )

    assert metrics.precision_at_k == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_relevant_item() -> None:
    calculator = RetrievalMetricCalculator()

    metrics = calculator.calculate(
        retrieved_chunk_ids=(
            "chunk-x",
            "chunk-a",
            "chunk-b",
        ),
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
                "chunk-b",
            }
        ),
        k=3,
    )

    assert metrics.reciprocal_rank == 0.5


def test_reciprocal_rank_is_zero_when_no_relevant_result() -> None:
    calculator = RetrievalMetricCalculator()

    metrics = calculator.calculate(
        retrieved_chunk_ids=(
            "chunk-x",
            "chunk-y",
        ),
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        k=2,
    )

    assert metrics.reciprocal_rank == 0.0


def test_k_limits_metrics() -> None:
    calculator = RetrievalMetricCalculator()

    metrics = calculator.calculate(
        retrieved_chunk_ids=(
            "chunk-x",
            "chunk-a",
        ),
        expected_relevant_chunk_ids=frozenset(
            {
                "chunk-a",
            }
        ),
        k=1,
    )

    assert metrics.hit_rate_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.reciprocal_rank == 0.0
