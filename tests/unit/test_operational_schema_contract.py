from agent_platform.observability.operational_snapshot import (
    OperationalObservabilitySnapshot,
)
from agent_platform.observability.operational_telemetry import (
    create_operational_telemetry_envelope,
)


def test_operational_event_schema_contract() -> None:
    snapshot = OperationalObservabilitySnapshot(
        health={},
        runtime={},
        api={},
        concurrency={},
        rate_limit={},
        security={},
        export={},
    )

    envelope = create_operational_telemetry_envelope(snapshot).to_dict()

    assert set(envelope) == {
        "schema_version",
        "event_type",
        "payload",
    }

    assert set(envelope["payload"]) == {
        "health",
        "runtime",
        "api",
        "concurrency",
        "rate_limit",
        "security",
        "export",
    }
