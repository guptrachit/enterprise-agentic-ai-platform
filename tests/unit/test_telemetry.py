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
        latency_ms=100.0,
        retry_count=0,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.0000125,
        prompt_name="ticket_classifier",
        prompt_version="1.0",
        workload="classification",
        logical_model="fast_general",
        fallback_used=True,
        fallback_from="classification_primary",
        fallback_reason="LLMTransientError",
        allowed_providers=("openai",),
        max_cost_tier="low",
        max_latency_tier="medium",
        prefer_lower_cost=True,
        prefer_lower_latency=True,
        preferred_providers=("openai", "anthropic"),
        preferred_cost_tier="low",
        preferred_latency_tier="fast",
    )

    assert event.provider == "openai"
    assert event.model == "gpt-5-mini"
    assert event.request_id == "resp_123"
    assert event.correlation_id == "corr-123"
    assert event.success is True
    assert event.latency_ms == 100.0
    assert event.retry_count == 0
    assert event.input_tokens == 10
    assert event.output_tokens == 5
    assert event.total_tokens == 15
    assert event.estimated_cost_usd == 0.0000125
    assert event.prompt_name == "ticket_classifier"
    assert event.prompt_version == "1.0"
    assert event.workload == "classification"
    assert event.logical_model == "fast_general"
    assert event.fallback_used is True
    assert event.fallback_from == "classification_primary"
    assert event.fallback_reason == "LLMTransientError"
    assert event.error_type is None
    assert event.allowed_providers == ("openai",)
    assert event.max_cost_tier == "low"
    assert event.max_latency_tier == "medium"
    assert event.prefer_lower_cost is True
    assert event.prefer_lower_latency is True
    assert event.preferred_providers == ("openai", "anthropic")
    assert event.preferred_cost_tier == "low"
    assert event.preferred_latency_tier == "fast"
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
        workload="classification",
        logical_model="fast_general",
        fallback_used=True,
        fallback_from="classification_primary",
        fallback_reason="LLMTransientError",
        allowed_providers=("openai",),
        max_cost_tier="low",
        max_latency_tier="medium",
        prefer_lower_cost=True,
        prefer_lower_latency=True,
        preferred_providers=("openai", "anthropic"),
        preferred_cost_tier="low",
        preferred_latency_tier="fast",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        log_execution_event(event)

    assert len(caplog.records) == 1

    message = caplog.records[0].getMessage()

    assert message.startswith("llm_execution ")

    payload = json.loads(message.removeprefix("llm_execution "))

    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-5-mini"
    assert payload["success"] is True
    assert payload["workload"] == "classification"
    assert payload["logical_model"] == "fast_general"
    assert payload["fallback_used"] is True
    assert payload["fallback_from"] == "classification_primary"
    assert payload["fallback_reason"] == "LLMTransientError"
    assert payload["retry_count"] == 0
    assert payload["total_tokens"] == 15
    assert payload["estimated_cost_usd"] == 0.0000125
    assert payload["allowed_providers"] == ["openai"]
    assert payload["max_cost_tier"] == "low"
    assert payload["max_latency_tier"] == "medium"
    assert payload["prefer_lower_cost"] is True
    assert payload["prefer_lower_latency"] is True
    assert payload["preferred_providers"] == ["openai", "anthropic"]
    assert payload["preferred_cost_tier"] == "low"
    assert payload["preferred_latency_tier"] == "fast"


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
    assert event.prefer_lower_cost is False
    assert event.prefer_lower_latency is False
    assert event.preferred_providers is None
    assert event.preferred_cost_tier is None
    assert event.preferred_latency_tier is None
