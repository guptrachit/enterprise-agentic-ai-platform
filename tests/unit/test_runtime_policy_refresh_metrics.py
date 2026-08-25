import pytest

from agent_platform.llm.runtime_policy_refresh_metrics import (
    RuntimePolicyRefreshMetrics,
)


def test_refresh_metrics_start_empty() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    snapshot = metrics.snapshot()

    assert snapshot.total_resolutions == 0
    assert snapshot.refreshes == 0
    assert snapshot.cache_hits == 0
    assert snapshot.policy_changes == 0

    assert snapshot.refresh_rate == 0.0
    assert snapshot.cache_hit_rate == 0.0
    assert snapshot.policy_change_rate == 0.0


def test_refresh_metrics_record_refresh() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    metrics.record_refresh(
        previous_policy_identifier=None,
        policy_identifier="production-policy@1.0.0",
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_resolutions == 1
    assert snapshot.refreshes == 1
    assert snapshot.cache_hits == 0
    assert snapshot.policy_changes == 0

    assert snapshot.refresh_rate == 1.0


def test_refresh_metrics_record_policy_change() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    metrics.record_refresh(
        previous_policy_identifier="production-policy@1.0.0",
        policy_identifier="production-policy@1.1.0",
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_resolutions == 1
    assert snapshot.refreshes == 1
    assert snapshot.policy_changes == 1
    assert snapshot.policy_change_rate == 1.0


def test_refresh_metrics_do_not_count_same_policy_as_change() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    metrics.record_refresh(
        previous_policy_identifier="production-policy@1.0.0",
        policy_identifier="production-policy@1.0.0",
    )

    snapshot = metrics.snapshot()

    assert snapshot.refreshes == 1
    assert snapshot.policy_changes == 0


def test_refresh_metrics_record_cache_hit() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    metrics.record_cache_hit()

    snapshot = metrics.snapshot()

    assert snapshot.total_resolutions == 1
    assert snapshot.refreshes == 0
    assert snapshot.cache_hits == 1

    assert snapshot.cache_hit_rate == 1.0


def test_refresh_metrics_calculate_rates() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    metrics.record_refresh(
        previous_policy_identifier=None,
        policy_identifier="production-policy@1.0.0",
    )

    metrics.record_cache_hit()

    metrics.record_cache_hit()

    metrics.record_refresh(
        previous_policy_identifier="production-policy@1.0.0",
        policy_identifier="production-policy@1.1.0",
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_resolutions == 4
    assert snapshot.refreshes == 2
    assert snapshot.cache_hits == 2
    assert snapshot.policy_changes == 1

    assert snapshot.refresh_rate == pytest.approx(0.5)

    assert snapshot.cache_hit_rate == pytest.approx(0.5)

    assert snapshot.policy_change_rate == pytest.approx(0.5)


def test_refresh_metrics_snapshot_to_dict() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    metrics.record_refresh(
        previous_policy_identifier="production-policy@1.0.0",
        policy_identifier="production-policy@1.1.0",
    )

    metrics.record_cache_hit()

    payload = metrics.snapshot().to_dict()

    assert payload == {
        "total_resolutions": 2,
        "refreshes": 1,
        "cache_hits": 1,
        "policy_changes": 1,
        "refresh_rate": 0.5,
        "cache_hit_rate": 0.5,
        "policy_change_rate": 1.0,
    }


def test_refresh_metrics_snapshots_are_point_in_time() -> None:
    metrics = RuntimePolicyRefreshMetrics()

    first = metrics.snapshot()

    metrics.record_cache_hit()

    second = metrics.snapshot()

    assert first.total_resolutions == 0
    assert second.total_resolutions == 1
