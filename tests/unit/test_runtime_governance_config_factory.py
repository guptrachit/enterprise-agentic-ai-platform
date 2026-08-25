import pytest

from agent_platform.config import Settings
from agent_platform.llm.runtime_governance_config_factory import (
    create_runtime_governance_config,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)


def test_settings_create_default_runtime_governance_config() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    config = create_runtime_governance_config(settings)

    assert config.policy_name == ("production-routing-policy")

    assert config.refresh.mode is RuntimePolicyRefreshMode.PROCESS_STATIC

    assert config.refresh.ttl_seconds is None
    assert config.enable_refresh_metrics is True


def test_settings_create_per_request_runtime_config() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_routing_policy_name="enterprise-policy",
        llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.PER_REQUEST),
    )

    config = create_runtime_governance_config(settings)

    assert config.policy_name == "enterprise-policy"

    assert config.refresh.mode is RuntimePolicyRefreshMode.PER_REQUEST


def test_settings_create_ttl_runtime_config() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.TTL),
        llm_routing_policy_ttl_seconds=120.0,
    )

    config = create_runtime_governance_config(settings)

    assert config.refresh.mode is RuntimePolicyRefreshMode.TTL

    assert config.refresh.ttl_seconds == 120.0


def test_settings_can_disable_refresh_metrics() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_routing_refresh_metrics_enabled=False,
    )

    config = create_runtime_governance_config(settings)

    assert config.enable_refresh_metrics is False


def test_settings_reject_ttl_without_ttl_seconds() -> None:
    with pytest.raises(
        ValueError,
        match="ttl_seconds is required",
    ):
        Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.TTL),
        )


def test_settings_reject_invalid_ttl_seconds() -> None:
    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.TTL),
            llm_routing_policy_ttl_seconds=0.0,
        )


def test_settings_reject_ttl_for_non_ttl_mode() -> None:
    with pytest.raises(
        ValueError,
        match="may only be configured",
    ):
        Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.PROCESS_STATIC),
            llm_routing_policy_ttl_seconds=60.0,
        )


def test_settings_reject_empty_policy_name() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        Settings(
            openai_api_key="test-key",
            llm_routing_policy_name=" ",
        )
