import logging
from unittest.mock import Mock

from agent_platform.observability.export_metrics import (
    ExportMetrics,
)
from agent_platform.observability.operational_telemetry import (
    OperationalTelemetryEnvelope,
)
from agent_platform.observability.resilient_exporter import (
    ResilientOperationalTelemetryExporter,
)


def create_envelope() -> OperationalTelemetryEnvelope:
    return OperationalTelemetryEnvelope(
        schema_version="1.0",
        event_type="operational_snapshot",
        payload={
            "health": {
                "status": "healthy",
            }
        },
    )


def test_resilient_exporter_delegates_successfully() -> None:
    child = Mock()
    metrics = ExportMetrics()

    exporter = ResilientOperationalTelemetryExporter(
        child,
        metrics=metrics,
    )

    envelope = create_envelope()

    exporter.export(envelope)

    child.export.assert_called_once_with(envelope)

    snapshot = metrics.snapshot()

    assert snapshot.total_exports == 1
    assert snapshot.successful_exports == 1
    assert snapshot.failed_exports == 0


def test_resilient_exporter_records_failure(
    caplog,
) -> None:
    child = Mock()

    child.export.side_effect = RuntimeError("backend unavailable")

    metrics = ExportMetrics()

    exporter = ResilientOperationalTelemetryExporter(
        child,
        metrics=metrics,
    )

    with caplog.at_level(
        logging.ERROR,
        logger="agent_platform.observability",
    ):
        exporter.export(create_envelope())

    snapshot = metrics.snapshot()

    assert snapshot.total_exports == 1
    assert snapshot.successful_exports == 0
    assert snapshot.failed_exports == 1

    records = [
        record
        for record in caplog.records
        if record.getMessage() == "operational_export_failed"
    ]

    assert len(records) == 1


def test_resilient_exporter_never_raises() -> None:
    child = Mock()

    child.export.side_effect = RuntimeError("backend unavailable")

    exporter = ResilientOperationalTelemetryExporter(child)

    exporter.export(create_envelope())


def test_resilient_exporter_exposes_metrics() -> None:
    child = Mock()

    metrics = ExportMetrics()

    exporter = ResilientOperationalTelemetryExporter(
        child,
        metrics=metrics,
    )

    assert exporter.metrics is metrics
