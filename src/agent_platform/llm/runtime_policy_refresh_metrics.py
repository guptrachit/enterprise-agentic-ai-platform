from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicyRefreshMetricsSnapshot:
    """Immutable runtime policy refresh metrics snapshot."""

    total_resolutions: int
    refreshes: int
    cache_hits: int
    policy_changes: int

    @property
    def refresh_rate(self) -> float:
        """Return the fraction of resolutions that performed a refresh."""

        if self.total_resolutions == 0:
            return 0.0

        return self.refreshes / self.total_resolutions

    @property
    def cache_hit_rate(self) -> float:
        """Return the fraction of resolutions that reused cached policy."""

        if self.total_resolutions == 0:
            return 0.0

        return self.cache_hits / self.total_resolutions

    @property
    def policy_change_rate(self) -> float:
        """Return the fraction of refreshes that changed policy version."""

        if self.refreshes == 0:
            return 0.0

        return self.policy_changes / self.refreshes

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable refresh metrics."""

        return {
            "total_resolutions": self.total_resolutions,
            "refreshes": self.refreshes,
            "cache_hits": self.cache_hits,
            "policy_changes": self.policy_changes,
            "refresh_rate": self.refresh_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "policy_change_rate": self.policy_change_rate,
        }


class RuntimePolicyRefreshMetrics:
    """In-memory counters for runtime policy refresh behavior."""

    def __init__(self) -> None:
        self._total_resolutions = 0
        self._refreshes = 0
        self._cache_hits = 0
        self._policy_changes = 0

    def record_refresh(
        self,
        *,
        previous_policy_identifier: str | None,
        policy_identifier: str,
    ) -> None:
        """Record one runtime policy refresh."""

        self._total_resolutions += 1
        self._refreshes += 1

        if (
            previous_policy_identifier is not None
            and previous_policy_identifier != policy_identifier
        ):
            self._policy_changes += 1

    def record_cache_hit(self) -> None:
        """Record one cached runtime policy reuse."""

        self._total_resolutions += 1
        self._cache_hits += 1

    def snapshot(
        self,
    ) -> RuntimePolicyRefreshMetricsSnapshot:
        """Return an immutable metrics snapshot."""

        return RuntimePolicyRefreshMetricsSnapshot(
            total_resolutions=self._total_resolutions,
            refreshes=self._refreshes,
            cache_hits=self._cache_hits,
            policy_changes=self._policy_changes,
        )
