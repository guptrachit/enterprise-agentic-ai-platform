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
    assert snapshot.failure_counts == {}
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
    assert snapshot.failure_counts == {}
    assert snapshot.average_latency_ms == 10.0
    assert snapshot.success_rate == 1.0


def test_api_metrics_record_failure_code() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=429,
        latency_ms=20.0,
        success=False,
        failure_code="llm_rate_limit_exceeded",
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_requests == 1
    assert snapshot.failed_requests == 1

    assert snapshot.status_counts == {
        429: 1,
    }

    assert snapshot.failure_counts == {
        "llm_rate_limit_exceeded": 1,
    }


def test_api_metrics_aggregate_failure_codes() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=429,
        latency_ms=10.0,
        success=False,
        failure_code="llm_rate_limit_exceeded",
    )

    metrics.record_request(
        status_code=429,
        latency_ms=20.0,
        success=False,
        failure_code="llm_rate_limit_exceeded",
    )

    metrics.record_request(
        status_code=504,
        latency_ms=30.0,
        success=False,
        failure_code="llm_request_timeout",
    )

    snapshot = metrics.snapshot()

    assert snapshot.failure_counts == {
        "llm_rate_limit_exceeded": 2,
        "llm_request_timeout": 1,
    }

    assert snapshot.failed_requests == 3


def test_api_metrics_do_not_record_failure_code_for_success() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=200,
        latency_ms=10.0,
        success=True,
        failure_code="should-not-count",
    )

    snapshot = metrics.snapshot()

    assert snapshot.failure_counts == {}


def test_api_metrics_support_failure_without_code() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=500,
        latency_ms=10.0,
        success=False,
    )

    snapshot = metrics.snapshot()

    assert snapshot.failed_requests == 1
    assert snapshot.failure_counts == {}


def test_api_metrics_aggregate_status_codes_and_rates() -> None:
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
        failure_code="llm_capacity_exceeded",
    )

    snapshot = metrics.snapshot()

    assert snapshot.status_counts == {
        200: 2,
        503: 1,
    }

    assert snapshot.failure_counts == {
        "llm_capacity_exceeded": 1,
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
        status_code=504,
        latency_ms=30.0,
        success=False,
        failure_code="llm_request_timeout",
    )

    payload = metrics.snapshot().to_dict()

    assert payload == {
        "total_requests": 2,
        "successful_requests": 1,
        "failed_requests": 1,
        "status_counts": {
            200: 1,
            504: 1,
        },
        "failure_counts": {
            "llm_request_timeout": 1,
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
        status_code=429,
        latency_ms=10.0,
        success=False,
        failure_code="llm_rate_limit_exceeded",
    )

    second = metrics.snapshot()

    assert first.total_requests == 0
    assert first.failure_counts == {}

    assert second.total_requests == 1

    assert second.failure_counts == {
        "llm_rate_limit_exceeded": 1,
    }
