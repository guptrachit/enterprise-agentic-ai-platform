from unittest.mock import Mock

from agent_platform.llm.api_guardrails import (
    InFlightRequestGuard,
)
from agent_platform.llm.api_health import (
    create_llm_api_health_payload,
)
from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)
from agent_platform.llm.api_rate_limit import (
    LLMAPIRateLimiter,
)
from agent_platform.security.security_metrics import (
    SecurityMetrics,
)


def create_runtime_mock(
    *,
    resolved: bool = True,
):
    runtime = Mock()

    health_snapshot = Mock()

    health_snapshot.to_dict.return_value = {
        "policy_name": "production-routing-policy",
        "policy_identifier": ("production-routing-policy@1.0.0" if resolved else None),
        "resolved": resolved,
    }

    runtime.health_snapshot.return_value = health_snapshot

    return runtime


def create_rate_limiter() -> LLMAPIRateLimiter:
    return LLMAPIRateLimiter(
        max_requests=60,
        window_seconds=60.0,
    )


def create_security_metrics() -> SecurityMetrics:
    return SecurityMetrics()


def test_llm_api_health_is_healthy() -> None:
    payload = create_llm_api_health_payload(
        runtime=create_runtime_mock(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=create_rate_limiter(),
        security_metrics=create_security_metrics(),
    )

    assert payload["status"] == "healthy"
    assert payload["reasons"] == []


def test_llm_api_health_is_degraded_when_runtime_unresolved() -> None:
    payload = create_llm_api_health_payload(
        runtime=create_runtime_mock(resolved=False),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=create_rate_limiter(),
        security_metrics=create_security_metrics(),
    )

    assert payload["status"] == "degraded"

    assert payload["reasons"] == [
        "routing_policy_unresolved",
    ]


def test_llm_api_health_is_degraded_after_timeout() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=504,
        latency_ms=10.0,
        success=False,
        failure_code="llm_request_timeout",
    )

    payload = create_llm_api_health_payload(
        runtime=create_runtime_mock(),
        api_metrics=metrics,
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=create_rate_limiter(),
        security_metrics=create_security_metrics(),
    )

    assert payload["status"] == "degraded"

    assert payload["reasons"] == [
        "llm_request_timeout",
    ]


def test_llm_api_health_contains_runtime_metrics_and_guardrails() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=200,
        latency_ms=10.0,
        success=True,
    )

    payload = create_llm_api_health_payload(
        runtime=create_runtime_mock(),
        api_metrics=metrics,
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=create_rate_limiter(),
        security_metrics=create_security_metrics(),
    )

    assert payload["runtime"]["policy_identifier"] == "production-routing-policy@1.0.0"

    assert payload["api_metrics"]["total_requests"] == 1

    assert payload["concurrency"] == {
        "max_in_flight": 10,
        "active_requests": 0,
        "available_capacity": 10,
        "capacity_rejections": 0,
        "utilization_rate": 0.0,
    }

    assert payload["rate_limit"] == {
        "max_requests": 60,
        "window_seconds": 60.0,
        "tracked_callers": 0,
        "rejected_requests": 0,
    }


def test_llm_api_health_exposes_failure_counts() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=429,
        latency_ms=10.0,
        success=False,
        failure_code="llm_rate_limit_exceeded",
    )

    payload = create_llm_api_health_payload(
        runtime=create_runtime_mock(),
        api_metrics=metrics,
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=create_rate_limiter(),
        security_metrics=create_security_metrics(),
    )

    assert payload["api_metrics"]["failure_counts"] == {
        "llm_rate_limit_exceeded": 1,
    }

    assert payload["status"] == "degraded"


def test_llm_api_health_exposes_security_metrics() -> None:
    security_metrics = SecurityMetrics()

    security_metrics.record_authentication_failure(
        failure_code="authentication_required"
    )

    security_metrics.record_authorization_failure(failure_code="authorization_denied")

    payload = create_llm_api_health_payload(
        runtime=create_runtime_mock(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=create_rate_limiter(),
        security_metrics=security_metrics,
    )

    assert payload["security"] == {
        "authentication_failures": 1,
        "authorization_failures": 1,
        "failure_counts": {
            "authentication_required": 1,
            "authorization_denied": 1,
        },
        "total_security_failures": 2,
    }


def test_llm_api_health_does_not_expose_sensitive_data() -> None:
    payload = create_llm_api_health_payload(
        runtime=create_runtime_mock(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=create_rate_limiter(),
        security_metrics=create_security_metrics(),
    )

    serialized = str(payload).lower()

    assert "api_key" not in serialized
    assert "openai_api_key" not in serialized
    assert "authorization:" not in serialized
    assert "bearer " not in serialized
    assert "bearer " not in serialized
