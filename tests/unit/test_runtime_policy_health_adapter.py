from unittest.mock import Mock

from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)
from agent_platform.llm.runtime_policy_health_adapter import (
    runtime_policy_health_payload,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshConfig,
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.runtime_policy_refresh_metrics import (
    RuntimePolicyRefreshMetrics,
)


def create_runtime_service(
    *,
    policy_identifier: str,
):
    service = Mock()
    service.policy_identifier = policy_identifier
    return service


def test_health_adapter_returns_process_static_payload() -> None:
    now = [100.0]

    def clock() -> float:
        return now[0]

    runtime_service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = runtime_service

    metrics = RuntimePolicyRefreshMetrics()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_metrics=metrics,
        clock=clock,
    )

    now[0] = 125.0

    payload = runtime_policy_health_payload(wrapper)

    assert payload["policy_name"] == ("production-policy")

    assert payload["policy_identifier"] == ("production-policy@1.0.0")

    assert payload["refresh_mode"] == ("process_static")

    assert payload["resolved"] is True
    assert payload["cache_age_seconds"] == 25.0

    assert payload["metrics"] == {
        "total_resolutions": 1,
        "refreshes": 1,
        "cache_hits": 0,
        "policy_changes": 0,
        "refresh_rate": 1.0,
        "cache_hit_rate": 0.0,
        "policy_change_rate": 0.0,
    }


def test_health_adapter_returns_unresolved_per_request_payload() -> None:
    factory = Mock()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
    )

    payload = runtime_policy_health_payload(wrapper)

    assert payload["policy_name"] == ("production-policy")

    assert payload["policy_identifier"] is None
    assert payload["refresh_mode"] == "per_request"
    assert payload["resolved"] is False
    assert payload["resolved_at"] is None
    assert payload["cache_age_seconds"] is None
    assert payload["metrics"] is None

    factory.create.assert_not_called()


def test_health_adapter_exposes_ttl_configuration() -> None:
    factory = Mock()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=60.0,
        ),
    )

    payload = runtime_policy_health_payload(wrapper)

    assert payload["refresh_mode"] == "ttl"
    assert payload["ttl_seconds"] == 60.0
    assert payload["resolved"] is False


def test_health_adapter_returns_independent_payload() -> None:
    runtime_service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = runtime_service

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
    )

    first = runtime_policy_health_payload(wrapper)

    first["policy_identifier"] = "mutated"

    second = runtime_policy_health_payload(wrapper)

    assert second["policy_identifier"] == ("production-policy@1.0.0")
