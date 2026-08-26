from unittest.mock import Mock

from agent_platform.observability.composite_exporter import (
    CompositeOperationalTelemetryExporter,
)
from agent_platform.observability.operational_telemetry import (
    OperationalTelemetryEnvelope,
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


def test_composite_exporter_calls_every_exporter() -> None:
    first = Mock()
    second = Mock()

    exporter = CompositeOperationalTelemetryExporter(
        exporters=[
            first,
            second,
        ]
    )

    envelope = create_envelope()

    exporter.export(envelope)

    first.export.assert_called_once_with(envelope)

    second.export.assert_called_once_with(envelope)


def test_composite_exporter_supports_single_exporter() -> None:
    child = Mock()

    exporter = CompositeOperationalTelemetryExporter(
        exporters=[
            child,
        ]
    )

    envelope = create_envelope()

    exporter.export(envelope)

    child.export.assert_called_once_with(envelope)


def test_composite_exporter_supports_empty_exporter_collection() -> None:
    exporter = CompositeOperationalTelemetryExporter(exporters=[])

    exporter.export(create_envelope())


def test_composite_exporter_propagates_export_failure() -> None:
    import pytest

    first = Mock()
    second = Mock()

    first.export.side_effect = RuntimeError("export failed")

    exporter = CompositeOperationalTelemetryExporter(
        exporters=[
            first,
            second,
        ]
    )

    envelope = create_envelope()

    with pytest.raises(
        RuntimeError,
        match="export failed",
    ):
        exporter.export(envelope)

    first.export.assert_called_once_with(envelope)

    second.export.assert_not_called()
