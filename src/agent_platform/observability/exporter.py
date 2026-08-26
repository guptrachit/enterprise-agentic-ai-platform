from abc import ABC, abstractmethod

from agent_platform.observability.operational_telemetry import (
    OperationalTelemetryEnvelope,
)


class OperationalTelemetryExporter(ABC):
    """Export versioned operational telemetry events."""

    @abstractmethod
    def export(
        self,
        envelope: OperationalTelemetryEnvelope,
    ) -> None:
        """Export one operational telemetry envelope."""
