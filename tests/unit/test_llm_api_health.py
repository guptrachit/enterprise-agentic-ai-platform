from unittest.mock import Mock

from agent_platform.llm.api_health import (
    create_llm_api_health_payload,
)
from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)


def test_llm_api_health_payload_contains_runtime_and_metrics() -> None:
    runtime = Mock()

    health_snapshot = Mock()

    health_snapshot.to_dict.return_value = {
        "policy_name": "production-routing-policy",
        "policy_identifier": ("production-routing-policy@1.0.0"),
        "resolved": True,
    }

    runtime.health_snapshot.return_value = health_snapshot

    metrics = LLMAPIMetrics()

    metrics.record_request(
        status_code=200,
        latency_ms=10.0,
        success=True,
    )

    payload = create_llm_api_health_payload(
        runtime=runtime,
        api_metrics=metrics,
    )

    assert payload["status"] == "healthy"

    assert payload["runtime"] == {
        "policy_name": "production-routing-policy",
        "policy_identifier": ("production-routing-policy@1.0.0"),
        "resolved": True,
    }

    api_metrics = payload["api_metrics"]

    assert isinstance(
        api_metrics,
        dict,
    )

    assert api_metrics["total_requests"] == 1
    assert api_metrics["successful_requests"] == 1


def test_llm_api_health_payload_does_not_expose_sensitive_data() -> None:
    runtime = Mock()

    health_snapshot = Mock()

    health_snapshot.to_dict.return_value = {
        "policy_name": "production-routing-policy",
        "policy_identifier": ("production-routing-policy@1.0.0"),
        "resolved": True,
    }

    runtime.health_snapshot.return_value = health_snapshot

    payload = create_llm_api_health_payload(
        runtime=runtime,
        api_metrics=LLMAPIMetrics(),
    )

    serialized = str(payload).lower()

    assert "api_key" not in serialized
    assert "openai_api_key" not in serialized
    assert "prompt" not in serialized
    assert "authorization" not in serialized
