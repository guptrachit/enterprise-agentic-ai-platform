import json
import logging

from agent_platform.llm.telemetry import (
    create_execution_event,
    log_execution_event,
)


def test_create_execution_event() -> None:
    event = create_execution_event(
        provider="openai",
        model="gpt-5-mini",
        request_id="resp_123",
        correlation_id="corr-123",
        success=True,
        latency_ms=123.4,
        retry_count=1,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.0000125,
    )

    assert event.provider == "openai"
    assert event.model == "gpt-5-mini"
    assert event.request_id == "resp_123"
    assert event.correlation_id == "corr-123"
    assert event.success is True
    assert event.latency_ms == 123.4
    assert event.retry_count == 1
    assert event.input_tokens == 10
    assert event.output_tokens == 5
    assert event.total_tokens == 15
    assert event.estimated_cost_usd == 0.0000125
    assert event.error_type is None
    assert event.timestamp


def test_log_execution_event(caplog) -> None:
    event = create_execution_event(
        provider="openai",
        model="gpt-5-mini",
        request_id="resp_123",
        correlation_id="corr-123",
        success=True,
        latency_ms=100.0,
        retry_count=0,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.0000125,
    )

    with caplog.at_level(logging.INFO, logger="agent_platform.llm"):
        log_execution_event(event)

    assert len(caplog.records) == 1

    message = caplog.records[0].getMessage()

    assert message.startswith("llm_execution ")

    payload = json.loads(message.removeprefix("llm_execution "))

    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-5-mini"
    assert payload["success"] is True
    assert payload["retry_count"] == 0
    assert payload["total_tokens"] == 15
    assert payload["estimated_cost_usd"] == 0.0000125


def test_failure_event_contains_error_type() -> None:
    event = create_execution_event(
        provider="openai",
        model="gpt-5-mini",
        request_id=None,
        correlation_id="corr-failure",
        success=False,
        latency_ms=250.0,
        retry_count=None,
        error_type="LLMRateLimitError",
    )

    assert event.success is False
    assert event.error_type == "LLMRateLimitError"
    assert event.input_tokens == 0
    assert event.output_tokens == 0
    assert event.total_tokens == 0
    assert event.estimated_cost_usd == 0.0
