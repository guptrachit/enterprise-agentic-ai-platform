import json
import logging

from agent_platform.observability.exporter import (
    OperationalTelemetryExporter,
)
from agent_platform.observability.operational_telemetry import (
    OperationalTelemetryEnvelope,
)

logger = logging.getLogger("agent_platform.observability")


class LoggingOperationalTelemetryExporter(OperationalTelemetryExporter):
    """Export operational telemetry through structured application logs."""

    def export(
        self,
        envelope: OperationalTelemetryEnvelope,
    ) -> None:
        """Log one structured operational telemetry event."""

        logger.info(
            "operational_snapshot %s",
            json.dumps(
                envelope.to_dict(),
                sort_keys=True,
            ),
        )
