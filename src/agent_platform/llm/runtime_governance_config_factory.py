from agent_platform.config import Settings
from agent_platform.llm.runtime_governance_config import (
    RuntimeGovernanceConfig,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshConfig,
)


def create_runtime_governance_config(
    settings: Settings,
) -> RuntimeGovernanceConfig:
    """Create governed runtime configuration from application settings."""

    return RuntimeGovernanceConfig(
        policy_name=settings.llm_routing_policy_name,
        refresh=RuntimePolicyRefreshConfig(
            mode=settings.llm_routing_policy_refresh_mode,
            ttl_seconds=settings.llm_routing_policy_ttl_seconds,
        ),
        enable_refresh_metrics=(settings.llm_routing_refresh_metrics_enabled),
    )
