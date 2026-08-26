from agent_platform.observability.operational_snapshot import (
    OperationalObservabilitySnapshot,
)
from agent_platform.observability.operational_telemetry import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OPERATIONAL_EVENT_TYPE,
    create_operational_telemetry_envelope,
)


def create_snapshot() -> OperationalObservabilitySnapshot:
    return OperationalObservabilitySnapshot(
        health={
            "status": "healthy",
            "reasons": [],
        },
        runtime={
            "resolved": True,
            "policy_identifier": ("production-routing-policy@1.0.0"),
        },
        api={
            "total_requests": 5,
            "successful_requests": 5,
            "failed_requests": 0,
        },
        concurrency={
            "max_in_flight": 100,
            "active_requests": 1,
        },
        rate_limit={
            "max_requests": 60,
            "rejected_requests": 0,
        },
        security={
            "authentication_failures": 0,
            "authorization_failures": 0,
            "total_security_failures": 0,
        },
        export={
            "total_exports": 0,
            "successful_exports": 0,
            "failed_exports": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
        },
    )


def test_operational_telemetry_envelope_has_schema_version() -> None:
    envelope = create_operational_telemetry_envelope(create_snapshot())

    assert envelope.schema_version == (OPERATIONAL_EVENT_SCHEMA_VERSION)

    assert envelope.schema_version == "1.0"

    assert envelope.event_type == (OPERATIONAL_EVENT_TYPE)

    assert envelope.event_type == ("operational_snapshot")


def test_operational_telemetry_envelope_contains_payload() -> None:
    envelope = create_operational_telemetry_envelope(create_snapshot())

    payload = envelope.to_dict()

    assert payload["schema_version"] == "1.0"

    assert payload["event_type"] == "operational_snapshot"

    assert payload["payload"]["health"]["status"] == "healthy"

    assert payload["payload"]["api"]["total_requests"] == 5


def test_operational_telemetry_envelope_does_not_expose_secrets() -> None:
    envelope = create_operational_telemetry_envelope(create_snapshot())

    serialized = str(envelope.to_dict()).lower()

    assert "openai_api_key" not in serialized
    assert "authorization:" not in serialized
    assert "bearer " not in serialized
    assert "api-key" not in serialized
    assert "prompt" not in serialized
