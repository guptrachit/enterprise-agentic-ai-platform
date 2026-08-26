from collections.abc import Iterable

from agent_platform.observability.exporter import (
    OperationalTelemetryExporter,
)
from agent_platform.observability.operational_telemetry import (
    OperationalTelemetryEnvelope,
)


class CompositeOperationalTelemetryExporter(OperationalTelemetryExporter):
    """Export one telemetry event to multiple exporters."""

    def __init__(
        self,
        exporters: Iterable[OperationalTelemetryExporter],
    ) -> None:
        self._exporters = tuple(exporters)

    def export(
        self,
        envelope: OperationalTelemetryEnvelope,
    ) -> None:
        """Export one envelope to every configured exporter."""

        for exporter in self._exporters:
            exporter.export(envelope)
