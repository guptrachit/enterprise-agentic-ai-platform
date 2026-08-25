from dataclasses import dataclass

from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshConfig,
    RuntimePolicyRefreshMode,
)


@dataclass(frozen=True)
class RuntimeGovernanceConfig:
    """Configuration for governed runtime routing behavior."""

    policy_name: str = "production-routing-policy"
    refresh: RuntimePolicyRefreshConfig = RuntimePolicyRefreshConfig()
    enable_refresh_metrics: bool = True

    def __post_init__(self) -> None:
        """Validate governed runtime configuration."""

        if not self.policy_name.strip():
            raise ValueError("runtime routing policy name must not be empty")

    @classmethod
    def process_static(
        cls,
        *,
        policy_name: str = "production-routing-policy",
        enable_refresh_metrics: bool = True,
    ) -> "RuntimeGovernanceConfig":
        """Create process-static runtime governance configuration."""

        return cls(
            policy_name=policy_name,
            refresh=RuntimePolicyRefreshConfig(
                mode=RuntimePolicyRefreshMode.PROCESS_STATIC,
            ),
            enable_refresh_metrics=enable_refresh_metrics,
        )

    @classmethod
    def per_request(
        cls,
        *,
        policy_name: str = "production-routing-policy",
        enable_refresh_metrics: bool = True,
    ) -> "RuntimeGovernanceConfig":
        """Create per-request runtime governance configuration."""

        return cls(
            policy_name=policy_name,
            refresh=RuntimePolicyRefreshConfig(
                mode=RuntimePolicyRefreshMode.PER_REQUEST,
            ),
            enable_refresh_metrics=enable_refresh_metrics,
        )

    @classmethod
    def ttl(
        cls,
        *,
        ttl_seconds: float,
        policy_name: str = "production-routing-policy",
        enable_refresh_metrics: bool = True,
    ) -> "RuntimeGovernanceConfig":
        """Create TTL-based runtime governance configuration."""

        return cls(
            policy_name=policy_name,
            refresh=RuntimePolicyRefreshConfig(
                mode=RuntimePolicyRefreshMode.TTL,
                ttl_seconds=ttl_seconds,
            ),
            enable_refresh_metrics=enable_refresh_metrics,
        )
