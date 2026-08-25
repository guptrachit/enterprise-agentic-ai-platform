import json
import logging

from agent_platform.llm.telemetry import (
    create_execution_event,
    create_routing_decision_event,
    log_execution_event,
    log_routing_decision_event,
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
        preferred_providers=(
            "openai",
            "anthropic",
        ),
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

    assert event.preferred_providers == (
        "openai",
        "anthropic",
    )

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
        preferred_providers=(
            "openai",
            "anthropic",
        ),
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
    assert payload["fallback_used"] is True
    assert payload["prefer_lower_cost"] is True
    assert payload["prefer_lower_latency"] is True

    assert payload["preferred_providers"] == [
        "openai",
        "anthropic",
    ]


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


def test_create_routing_decision_event() -> None:
    event = create_routing_decision_event(
        workload="classification",
        selected_model="cheap_model",
        ranked_candidates=(
            "cheap_model",
            "backup_model",
        ),
        rejected_models=("disabled_model",),
        routing_reason_codes=(
            "disabled",
            "lower_cost_preferred",
            "selected",
        ),
        routing_reasons=(
            "disabled_model rejected because it is disabled",
            "lower cost preferred",
            "cheap_model selected",
        ),
        executed_model="cheap_model",
        fallback_used=False,
        success=True,
        correlation_id="corr-routing-123",
    )

    assert event.workload == "classification"
    assert event.selected_model == "cheap_model"

    assert event.ranked_candidates == (
        "cheap_model",
        "backup_model",
    )

    assert event.rejected_models == ("disabled_model",)

    assert event.routing_reason_codes == (
        "disabled",
        "lower_cost_preferred",
        "selected",
    )

    assert event.routing_reasons == (
        "disabled_model rejected because it is disabled",
        "lower cost preferred",
        "cheap_model selected",
    )

    assert event.executed_model == "cheap_model"
    assert event.fallback_used is False
    assert event.success is True
    assert event.correlation_id == "corr-routing-123"
    assert event.error_type is None
    assert event.timestamp


def test_log_routing_decision_event(caplog) -> None:
    event = create_routing_decision_event(
        workload="general",
        selected_model="primary",
        ranked_candidates=(
            "primary",
            "backup",
        ),
        rejected_models=(),
        routing_reason_codes=("selected",),
        routing_reasons=("primary ranked first",),
        executed_model="backup",
        fallback_used=True,
        success=True,
        correlation_id="corr-routing-456",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        log_routing_decision_event(event)

    assert len(caplog.records) == 1

    message = caplog.records[0].getMessage()

    assert message.startswith("llm_routing_decision ")

    payload = json.loads(message.removeprefix("llm_routing_decision "))

    assert payload["selected_model"] == "primary"

    assert payload["ranked_candidates"] == [
        "primary",
        "backup",
    ]

    assert payload["routing_reason_codes"] == [
        "selected",
    ]

    assert payload["routing_reasons"] == [
        "primary ranked first",
    ]

    assert payload["executed_model"] == "backup"
    assert payload["fallback_used"] is True
    assert payload["success"] is True


def test_routing_decision_failure_event() -> None:
    event = create_routing_decision_event(
        workload="general",
        selected_model="primary",
        ranked_candidates=("primary",),
        rejected_models=(),
        routing_reason_codes=("selected",),
        routing_reasons=("primary selected",),
        executed_model="primary",
        fallback_used=False,
        success=False,
        error_type="LLMInvalidRequestError",
    )

    assert event.success is False
    assert event.error_type == "LLMInvalidRequestError"

    assert event.routing_reason_codes == ("selected",)


def test_routing_decision_event_preserves_policy_identifier() -> None:
    event = create_routing_decision_event(
        workload="general",
        selected_model="primary",
        ranked_candidates=("primary",),
        rejected_models=(),
        routing_reason_codes=("selected",),
        routing_reasons=("Selected model 'primary'",),
        executed_model="primary",
        fallback_used=False,
        success=True,
        correlation_id="corr-001",
        policy_identifier=("production-routing-policy@1.2.0"),
    )

    assert event.policy_identifier == ("production-routing-policy@1.2.0")
