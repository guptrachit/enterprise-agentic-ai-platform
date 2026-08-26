import json
import logging

from agent_platform.llm.api_telemetry import (
    create_llm_api_request_event,
    log_llm_api_request_event,
)


def test_api_telemetry_preserves_success_values() -> None:
    event = create_llm_api_request_event(
        correlation_id="corr-001",
        status_code=200,
        latency_ms=12.5,
        success=True,
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    assert event.correlation_id == "corr-001"
    assert event.status_code == 200
    assert event.latency_ms == 12.5
    assert event.success is True

    assert event.policy_identifier == ("production-routing-policy@1.0.0")

    assert event.model == "primary"
    assert event.provider == "openai"
    assert event.failure_code is None
    assert event.timestamp


def test_api_telemetry_preserves_failure_code() -> None:
    event = create_llm_api_request_event(
        correlation_id="corr-002",
        status_code=429,
        latency_ms=20.0,
        success=False,
        policy_identifier=None,
        model=None,
        provider=None,
        failure_code="llm_rate_limit_exceeded",
    )

    assert event.success is False
    assert event.status_code == 429

    assert event.failure_code == ("llm_rate_limit_exceeded")


def test_api_telemetry_logging(caplog) -> None:
    event = create_llm_api_request_event(
        correlation_id="corr-003",
        status_code=503,
        latency_ms=7.5,
        success=False,
        policy_identifier=None,
        model=None,
        provider=None,
        failure_code="llm_capacity_exceeded",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        log_llm_api_request_event(event)

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_api_request ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("llm_api_request "))

    assert payload["correlation_id"] == "corr-003"
    assert payload["status_code"] == 503
    assert payload["success"] is False

    assert payload["failure_code"] == ("llm_capacity_exceeded")
