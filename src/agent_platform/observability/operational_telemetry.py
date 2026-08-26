from dataclasses import dataclass

from agent_platform.observability.operational_snapshot import (
    OperationalObservabilitySnapshot,
)

OPERATIONAL_EVENT_SCHEMA_VERSION = "1.0"
OPERATIONAL_EVENT_TYPE = "operational_snapshot"


@dataclass(frozen=True)
class OperationalTelemetryEnvelope:
    """Versioned structured observability event envelope."""

    schema_version: str
    event_type: str
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable event envelope."""

        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "payload": dict(self.payload),
        }


def create_operational_telemetry_envelope(
    snapshot: OperationalObservabilitySnapshot,
) -> OperationalTelemetryEnvelope:
    """Create a versioned operational telemetry event."""

    return OperationalTelemetryEnvelope(
        schema_version=OPERATIONAL_EVENT_SCHEMA_VERSION,
        event_type=OPERATIONAL_EVENT_TYPE,
        payload=snapshot.to_dict(),
    )
