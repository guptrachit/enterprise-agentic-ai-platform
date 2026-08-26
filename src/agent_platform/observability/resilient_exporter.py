import logging

from agent_platform.observability.export_metrics import (
    ExportMetrics,
)
from agent_platform.observability.exporter import (
    OperationalTelemetryExporter,
)
from agent_platform.observability.operational_telemetry import (
    OperationalTelemetryEnvelope,
)

logger = logging.getLogger("agent_platform.observability")


class ResilientOperationalTelemetryExporter(OperationalTelemetryExporter):
    """Isolate application execution from exporter failures."""

    def __init__(
        self,
        exporter: OperationalTelemetryExporter,
        *,
        metrics: ExportMetrics | None = None,
    ) -> None:
        self._exporter = exporter
        self._metrics = metrics or ExportMetrics()

    @property
    def metrics(self) -> ExportMetrics:
        """Return exporter reliability metrics."""

        return self._metrics

    def export(
        self,
        envelope: OperationalTelemetryEnvelope,
    ) -> None:
        """Export telemetry without propagating failures."""

        try:
            self._exporter.export(envelope)
        except Exception:
            self._metrics.record_failure()

            logger.exception("operational_export_failed")

            return

        self._metrics.record_success()
