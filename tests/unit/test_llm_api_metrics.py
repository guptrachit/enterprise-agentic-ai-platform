import pytest

from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)


def test_api_metrics_start_empty() -> None:
    metrics = LLMAPIMetrics()

    snapshot = metrics.snapshot()

    assert snapshot.total_requests == 0
    assert snapshot.successful_requests == 0
    assert snapshot.failed_requests == 0
    assert snapshot.status_counts == {}
    assert snapshot.total_latency_ms == 0.0
    assert snapshot.average_latency_ms == 0.0
    assert snapshot.success_rate == 0.0
    assert snapshot.failure_rate == 0.0


def test_api_metrics_record_success() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=200,
        latency_ms=10.0,
        success=True,
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_requests == 1
    assert snapshot.successful_requests == 1
    assert snapshot.failed_requests == 0
    assert snapshot.status_counts == {200: 1}
    assert snapshot.average_latency_ms == 10.0
    assert snapshot.success_rate == 1.0


def test_api_metrics_record_failure() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=503,
        latency_ms=20.0,
        success=False,
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_requests == 1
    assert snapshot.failed_requests == 1
    assert snapshot.status_counts == {503: 1}
    assert snapshot.failure_rate == 1.0


def test_api_metrics_aggregate_status_codes() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=200,
        latency_ms=10.0,
        success=True,
    )

    metrics.record_request(
        status_code=200,
        latency_ms=20.0,
        success=True,
    )

    metrics.record_request(
        status_code=503,
        latency_ms=30.0,
        success=False,
    )

    snapshot = metrics.snapshot()

    assert snapshot.status_counts == {
        200: 2,
        503: 1,
    }

    assert snapshot.total_requests == 3
    assert snapshot.successful_requests == 2
    assert snapshot.failed_requests == 1

    assert snapshot.average_latency_ms == pytest.approx(20.0)

    assert snapshot.success_rate == pytest.approx(2 / 3)

    assert snapshot.failure_rate == pytest.approx(1 / 3)


def test_api_metrics_snapshot_to_dict() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=200,
        latency_ms=10.0,
        success=True,
    )

    metrics.record_request(
        status_code=500,
        latency_ms=30.0,
        success=False,
    )

    payload = metrics.snapshot().to_dict()

    assert payload == {
        "total_requests": 2,
        "successful_requests": 1,
        "failed_requests": 1,
        "status_counts": {
            200: 1,
            500: 1,
        },
        "total_latency_ms": 40.0,
        "average_latency_ms": 20.0,
        "success_rate": 0.5,
        "failure_rate": 0.5,
    }


def test_api_metrics_snapshot_is_point_in_time() -> None:
    metrics = LLMAPIMetrics()

    first = metrics.snapshot()

    metrics.record_request(
        status_code=200,
        latency_ms=10.0,
        success=True,
    )

    second = metrics.snapshot()

    assert first.total_requests == 0
    assert second.total_requests == 1
