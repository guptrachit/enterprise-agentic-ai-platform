from agent_platform.llm.routing_health import (
    RoutingHealthStatus,
    evaluate_routing_health,
)
from agent_platform.llm.routing_metrics import RoutingMetrics


def test_empty_metrics_are_healthy() -> None:
    metrics = RoutingMetrics()

    health = evaluate_routing_health(metrics.snapshot())

    assert health.status is RoutingHealthStatus.HEALTHY
    assert health.total_requests == 0
    assert health.failure_rate == 0.0
    assert health.fallback_rate == 0.0
    assert health.metrics_export_failures == 0
    assert health.reasons == ()


def test_successful_routing_is_healthy() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    health = evaluate_routing_health(metrics.snapshot())

    assert health.status is RoutingHealthStatus.HEALTHY
    assert health.reasons == ()


def test_failure_below_fifty_percent_is_degraded() -> None:
    metrics = RoutingMetrics()

    for success in (
        True,
        True,
        False,
    ):
        metrics.record_request(
            workload="general",
            selected_model="primary",
            executed_model="primary",
            routing_reason_codes=("selected",),
            success=success,
            fallback_used=False,
        )

    health = evaluate_routing_health(metrics.snapshot())

    assert health.status is RoutingHealthStatus.DEGRADED

    assert "One or more routing requests failed." in health.reasons


def test_failure_rate_at_fifty_percent_is_unhealthy() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=False,
        fallback_used=False,
    )

    health = evaluate_routing_health(metrics.snapshot())

    assert health.status is RoutingHealthStatus.UNHEALTHY
    assert health.failure_rate == 0.5


def test_high_fallback_rate_is_degraded() -> None:
    metrics = RoutingMetrics()

    for fallback_used in (
        True,
        False,
        False,
        False,
    ):
        metrics.record_request(
            workload="general",
            selected_model="primary",
            executed_model=("backup" if fallback_used else "primary"),
            routing_reason_codes=("selected",),
            success=True,
            fallback_used=fallback_used,
        )

    health = evaluate_routing_health(metrics.snapshot())

    assert health.status is RoutingHealthStatus.DEGRADED
    assert health.fallback_rate == 0.25


def test_export_failure_is_degraded() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    metrics.record_export_failure()

    health = evaluate_routing_health(metrics.snapshot())

    assert health.status is RoutingHealthStatus.DEGRADED

    assert "One or more routing metrics exports failed." in health.reasons


def test_unhealthy_failure_rate_takes_priority() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=("selected",),
        success=False,
        fallback_used=True,
    )

    metrics.record_export_failure()

    health = evaluate_routing_health(metrics.snapshot())

    assert health.status is RoutingHealthStatus.UNHEALTHY

    assert health.reasons == ("Routing failure rate is at or above 50%.",)
