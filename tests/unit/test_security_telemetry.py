import json
import logging

from agent_platform.security.security_telemetry import (
    SecurityEventType,
    create_security_event,
    log_security_event,
)


def test_security_event_preserves_safe_values() -> None:
    event = create_security_event(
        correlation_id="corr-security-001",
        event_type=(SecurityEventType.AUTHORIZATION_FAILURE),
        status_code=403,
        failure_code="authorization_denied",
        path="/llm/generate",
    )

    assert event.correlation_id == "corr-security-001"

    assert event.event_type is SecurityEventType.AUTHORIZATION_FAILURE

    assert event.status_code == 403
    assert event.failure_code == "authorization_denied"
    assert event.path == "/llm/generate"
    assert event.timestamp


def test_security_event_logging(caplog) -> None:
    event = create_security_event(
        correlation_id="corr-security-002",
        event_type=(SecurityEventType.AUTHENTICATION_FAILURE),
        status_code=401,
        failure_code="authentication_required",
        path="/llm/generate",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.security",
    ):
        log_security_event(event)

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("security_event ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("security_event "))

    assert payload["correlation_id"] == ("corr-security-002")

    assert payload["event_type"] == ("authentication_failure")

    assert payload["status_code"] == 401

    assert payload["failure_code"] == ("authentication_required")

    assert payload["path"] == "/llm/generate"


def test_security_event_does_not_contain_sensitive_fields(
    caplog,
) -> None:
    event = create_security_event(
        correlation_id="corr-security-003",
        event_type=(SecurityEventType.AUTHORIZATION_FAILURE),
        status_code=403,
        failure_code="authorization_denied",
        path="/llm/generate",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.security",
    ):
        log_security_event(event)

    serialized = "\n".join(record.getMessage() for record in caplog.records).lower()

    assert "authorization:" not in serialized
    assert "bearer " not in serialized
    assert "api_key" not in serialized
    assert "prompt" not in serialized
