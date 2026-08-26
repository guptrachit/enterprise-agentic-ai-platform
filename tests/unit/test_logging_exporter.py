import json
import logging

from agent_platform.observability.logging_exporter import (
    LoggingOperationalTelemetryExporter,
)
from agent_platform.observability.operational_telemetry import (
    OperationalTelemetryEnvelope,
)


def test_logging_exporter_logs_envelope(
    caplog,
) -> None:
    exporter = LoggingOperationalTelemetryExporter()

    envelope = OperationalTelemetryEnvelope(
        schema_version="1.0",
        event_type="operational_snapshot",
        payload={
            "health": {
                "status": "healthy",
            }
        },
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.observability",
    ):
        exporter.export(envelope)

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("operational_snapshot ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("operational_snapshot "))

    assert payload == {
        "schema_version": "1.0",
        "event_type": "operational_snapshot",
        "payload": {
            "health": {
                "status": "healthy",
            }
        },
    }
