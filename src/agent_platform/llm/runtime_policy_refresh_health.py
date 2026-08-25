from dataclasses import dataclass

from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.runtime_policy_refresh_metrics import (
    RuntimePolicyRefreshMetricsSnapshot,
)


@dataclass(frozen=True)
class RuntimePolicyRefreshHealthSnapshot:
    """Operational health snapshot for runtime policy refresh state."""

    policy_name: str
    policy_identifier: str | None
    refresh_mode: RuntimePolicyRefreshMode
    ttl_seconds: float | None
    resolved_at: float | None
    cache_age_seconds: float | None
    metrics: RuntimePolicyRefreshMetricsSnapshot | None

    @property
    def resolved(self) -> bool:
        """Return whether a policy has currently been resolved."""

        return self.policy_identifier is not None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable health snapshot."""

        return {
            "policy_name": self.policy_name,
            "policy_identifier": self.policy_identifier,
            "refresh_mode": self.refresh_mode.value,
            "ttl_seconds": self.ttl_seconds,
            "resolved_at": self.resolved_at,
            "cache_age_seconds": self.cache_age_seconds,
            "resolved": self.resolved,
            "metrics": (self.metrics.to_dict() if self.metrics is not None else None),
        }
