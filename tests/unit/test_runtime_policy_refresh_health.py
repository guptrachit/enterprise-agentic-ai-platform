from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.runtime_policy_refresh_health import (
    RuntimePolicyRefreshHealthSnapshot,
)
from agent_platform.llm.runtime_policy_refresh_metrics import (
    RuntimePolicyRefreshMetricsSnapshot,
)


def test_health_snapshot_preserves_values() -> None:
    metrics = RuntimePolicyRefreshMetricsSnapshot(
        total_resolutions=4,
        refreshes=2,
        cache_hits=2,
        policy_changes=1,
    )

    snapshot = RuntimePolicyRefreshHealthSnapshot(
        policy_name="production-policy",
        policy_identifier="production-policy@1.1.0",
        refresh_mode=RuntimePolicyRefreshMode.TTL,
        ttl_seconds=60.0,
        resolved_at=100.0,
        cache_age_seconds=30.0,
        metrics=metrics,
    )

    assert snapshot.policy_name == "production-policy"

    assert snapshot.policy_identifier == ("production-policy@1.1.0")

    assert snapshot.refresh_mode is RuntimePolicyRefreshMode.TTL
    assert snapshot.ttl_seconds == 60.0
    assert snapshot.resolved_at == 100.0
    assert snapshot.cache_age_seconds == 30.0
    assert snapshot.resolved is True


def test_unresolved_health_snapshot() -> None:
    snapshot = RuntimePolicyRefreshHealthSnapshot(
        policy_name="production-policy",
        policy_identifier=None,
        refresh_mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ttl_seconds=None,
        resolved_at=None,
        cache_age_seconds=None,
        metrics=None,
    )

    assert snapshot.resolved is False


def test_health_snapshot_to_dict() -> None:
    metrics = RuntimePolicyRefreshMetricsSnapshot(
        total_resolutions=2,
        refreshes=1,
        cache_hits=1,
        policy_changes=0,
    )

    snapshot = RuntimePolicyRefreshHealthSnapshot(
        policy_name="production-policy",
        policy_identifier="production-policy@1.0.0",
        refresh_mode=RuntimePolicyRefreshMode.TTL,
        ttl_seconds=60.0,
        resolved_at=100.0,
        cache_age_seconds=20.0,
        metrics=metrics,
    )

    payload = snapshot.to_dict()

    assert payload == {
        "policy_name": "production-policy",
        "policy_identifier": ("production-policy@1.0.0"),
        "refresh_mode": "ttl",
        "ttl_seconds": 60.0,
        "resolved_at": 100.0,
        "cache_age_seconds": 20.0,
        "resolved": True,
        "metrics": {
            "total_resolutions": 2,
            "refreshes": 1,
            "cache_hits": 1,
            "policy_changes": 0,
            "refresh_rate": 0.5,
            "cache_hit_rate": 0.5,
            "policy_change_rate": 0.0,
        },
    }


def test_health_snapshot_without_metrics_serializes_none() -> None:
    snapshot = RuntimePolicyRefreshHealthSnapshot(
        policy_name="production-policy",
        policy_identifier=None,
        refresh_mode=(RuntimePolicyRefreshMode.PROCESS_STATIC),
        ttl_seconds=None,
        resolved_at=None,
        cache_age_seconds=None,
        metrics=None,
    )

    payload = snapshot.to_dict()

    assert payload["metrics"] is None
    assert payload["resolved"] is False
