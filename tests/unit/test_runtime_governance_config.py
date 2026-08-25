import pytest

from agent_platform.llm.runtime_governance_config import (
    RuntimeGovernanceConfig,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)


def test_runtime_governance_defaults() -> None:
    config = RuntimeGovernanceConfig()

    assert config.policy_name == ("production-routing-policy")

    assert config.refresh.mode is RuntimePolicyRefreshMode.PROCESS_STATIC

    assert config.refresh.ttl_seconds is None
    assert config.enable_refresh_metrics is True


@pytest.mark.parametrize(
    "policy_name",
    (
        "",
        " ",
        "   ",
    ),
)
def test_runtime_governance_rejects_empty_policy_name(
    policy_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        RuntimeGovernanceConfig(
            policy_name=policy_name,
        )


def test_process_static_factory() -> None:
    config = RuntimeGovernanceConfig.process_static(
        policy_name="enterprise-policy",
    )

    assert config.policy_name == "enterprise-policy"

    assert config.refresh.mode is RuntimePolicyRefreshMode.PROCESS_STATIC

    assert config.refresh.ttl_seconds is None


def test_per_request_factory() -> None:
    config = RuntimeGovernanceConfig.per_request(
        policy_name="enterprise-policy",
    )

    assert config.policy_name == "enterprise-policy"

    assert config.refresh.mode is RuntimePolicyRefreshMode.PER_REQUEST


def test_ttl_factory() -> None:
    config = RuntimeGovernanceConfig.ttl(
        policy_name="enterprise-policy",
        ttl_seconds=120.0,
    )

    assert config.policy_name == "enterprise-policy"

    assert config.refresh.mode is RuntimePolicyRefreshMode.TTL
    assert config.refresh.ttl_seconds == 120.0


def test_ttl_factory_preserves_refresh_metrics_setting() -> None:
    config = RuntimeGovernanceConfig.ttl(
        ttl_seconds=60.0,
        enable_refresh_metrics=False,
    )

    assert config.enable_refresh_metrics is False


def test_ttl_factory_rejects_invalid_ttl() -> None:
    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        RuntimeGovernanceConfig.ttl(
            ttl_seconds=0.0,
        )


def test_process_static_can_disable_refresh_metrics() -> None:
    config = RuntimeGovernanceConfig.process_static(
        enable_refresh_metrics=False,
    )

    assert config.enable_refresh_metrics is False


def test_per_request_can_disable_refresh_metrics() -> None:
    config = RuntimeGovernanceConfig.per_request(
        enable_refresh_metrics=False,
    )

    assert config.enable_refresh_metrics is False
