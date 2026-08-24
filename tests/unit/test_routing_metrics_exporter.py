import json
import logging

from agent_platform.llm.routing_metrics import RoutingMetrics
from agent_platform.llm.routing_metrics_exporter import (
    LoggingRoutingMetricsExporter,
    RoutingMetricsExporter,
)


def test_logging_exporter_matches_exporter_protocol() -> None:
    exporter: RoutingMetricsExporter = LoggingRoutingMetricsExporter()

    assert isinstance(
        exporter,
        LoggingRoutingMetricsExporter,
    )


def test_logging_exporter_writes_structured_metrics(
    caplog,
) -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="classification",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=(
            "constraint_rejected",
            "selected",
        ),
        success=True,
        fallback_used=True,
    )

    exporter = LoggingRoutingMetricsExporter()

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        exporter.export(metrics.snapshot())

    assert len(caplog.records) == 1

    message = caplog.records[0].getMessage()

    assert message.startswith("llm_routing_metrics ")

    payload = json.loads(message.removeprefix("llm_routing_metrics "))

    assert payload["total_requests"] == 1
    assert payload["successful_requests"] == 1
    assert payload["failed_requests"] == 0
    assert payload["fallback_requests"] == 1
    assert payload["success_rate"] == 1.0
    assert payload["failure_rate"] == 0.0
    assert payload["fallback_rate"] == 1.0

    assert payload["model_selection_counts"] == {
        "primary": 1,
    }

    assert payload["executed_model_counts"] == {
        "backup": 1,
    }

    assert payload["rejection_reason_counts"] == {
        "constraint_rejected": 1,
        "selected": 1,
    }

    assert payload["workload_counts"] == {
        "classification": 1,
    }


def test_logging_exporter_does_not_mutate_snapshot() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    snapshot = metrics.snapshot()

    before = snapshot.to_dict()

    exporter = LoggingRoutingMetricsExporter()
    exporter.export(snapshot)

    after = snapshot.to_dict()

    assert after == before
