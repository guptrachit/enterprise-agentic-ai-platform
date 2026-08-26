from unittest.mock import Mock

from agent_platform.llm.api_guardrails import (
    InFlightRequestGuard,
)
from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)
from agent_platform.llm.api_rate_limit import (
    LLMAPIRateLimiter,
)
from agent_platform.observability.operational_service import (
    OperationalObservabilityService,
)
from agent_platform.security.security_metrics import (
    SecurityMetrics,
)


def create_runtime(
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


def test_operational_service_builds_healthy_snapshot() -> None:
    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
    )

    snapshot = service.snapshot()

    assert snapshot.health == {
        "status": "healthy",
        "reasons": [],
    }

    assert snapshot.runtime["resolved"] is True

    assert snapshot.api["total_requests"] == 0

    assert snapshot.concurrency["max_in_flight"] == 10

    assert snapshot.rate_limit["max_requests"] == 60

    assert snapshot.security["total_security_failures"] == 0


def test_operational_service_reports_degraded_runtime() -> None:
    service = OperationalObservabilityService(
        runtime=create_runtime(resolved=False),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
    )

    snapshot = service.snapshot()

    assert snapshot.health["status"] == "degraded"

    assert snapshot.health["reasons"] == [
        "routing_policy_unresolved",
    ]


def test_operational_service_includes_api_failures() -> None:
    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=504,
        latency_ms=10.0,
        success=False,
        failure_code="llm_request_timeout",
    )

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=metrics,
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
    )

    snapshot = service.snapshot()

    assert snapshot.health["status"] == "degraded"

    assert snapshot.health["reasons"] == [
        "llm_request_timeout",
    ]

    assert snapshot.api["failure_counts"] == {
        "llm_request_timeout": 1,
    }


def test_operational_service_exports_snapshot_with_default_exporter(
    caplog,
) -> None:
    import logging

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.observability",
    ):
        snapshot = service.export_snapshot()

    assert snapshot.health["status"] == "healthy"

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("operational_snapshot ")
    ]

    assert len(records) == 1


def test_operational_service_exports_snapshot() -> None:
    exporter = Mock()

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    snapshot = service.export_snapshot()

    assert snapshot.health["status"] == "healthy"

    exporter.export.assert_called_once()

    envelope = exporter.export.call_args.args[0]

    assert envelope.schema_version == "1.0"

    assert envelope.event_type == ("operational_snapshot")

    assert envelope.payload["health"]["status"] == "healthy"


def test_operational_service_supports_composite_exporter() -> None:
    from unittest.mock import Mock

    from agent_platform.observability.composite_exporter import (
        CompositeOperationalTelemetryExporter,
    )

    first = Mock()
    second = Mock()

    exporter = CompositeOperationalTelemetryExporter(
        exporters=[
            first,
            second,
        ]
    )

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    snapshot = service.export_snapshot()

    assert snapshot.health["status"] == "healthy"

    first.export.assert_called_once()
    second.export.assert_called_once()

    first_envelope = first.export.call_args.args[0]

    second_envelope = second.export.call_args.args[0]

    assert first_envelope == second_envelope


def test_operational_service_isolates_export_failure() -> None:
    exporter = Mock()

    exporter.export.side_effect = RuntimeError("backend unavailable")

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    snapshot = service.export_snapshot()

    assert snapshot.health["status"] == "healthy"

    exporter.export.assert_called_once()


def test_operational_service_records_successful_export_metric() -> None:
    exporter = Mock()

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    service.export_snapshot()

    metrics = service.export_metrics.snapshot()

    assert metrics.total_exports == 1
    assert metrics.successful_exports == 1
    assert metrics.failed_exports == 0


def test_operational_service_records_failed_export_metric() -> None:
    exporter = Mock()

    exporter.export.side_effect = RuntimeError("backend unavailable")

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    service.export_snapshot()

    metrics = service.export_metrics.snapshot()

    assert metrics.total_exports == 1
    assert metrics.successful_exports == 0
    assert metrics.failed_exports == 1


def test_operational_snapshot_contains_export_metrics() -> None:
    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
    )

    snapshot = service.snapshot()

    assert snapshot.export == {
        "total_exports": 0,
        "successful_exports": 0,
        "failed_exports": 0,
        "success_rate": 0.0,
        "failure_rate": 0.0,
    }


def test_operational_snapshot_reflects_export_history() -> None:
    exporter = Mock()

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    service.export_snapshot()

    snapshot = service.snapshot()

    assert snapshot.export["total_exports"] == 1

    assert snapshot.export["successful_exports"] == 1

    assert snapshot.export["failed_exports"] == 0


def test_operational_service_degrades_after_export_failure() -> None:
    exporter = Mock()

    exporter.export.side_effect = RuntimeError("backend unavailable")

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    first_snapshot = service.export_snapshot()

    assert first_snapshot.health["status"] == "healthy"

    second_snapshot = service.snapshot()

    assert second_snapshot.health["status"] == "degraded"

    assert second_snapshot.health["reasons"] == [
        "telemetry_export_degraded",
    ]

    assert second_snapshot.export["failed_exports"] == 1


def test_operational_service_remains_healthy_after_successful_export() -> None:
    exporter = Mock()

    service = OperationalObservabilityService(
        runtime=create_runtime(),
        api_metrics=LLMAPIMetrics(),
        concurrency_guard=InFlightRequestGuard(max_in_flight=10),
        rate_limiter=LLMAPIRateLimiter(
            max_requests=60,
            window_seconds=60.0,
        ),
        security_metrics=SecurityMetrics(),
        exporter=exporter,
    )

    service.export_snapshot()

    snapshot = service.snapshot()

    assert snapshot.health["status"] == "healthy"

    assert snapshot.health["reasons"] == []

    assert snapshot.export["successful_exports"] == 1
