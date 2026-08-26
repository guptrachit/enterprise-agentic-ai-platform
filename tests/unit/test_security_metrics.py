from agent_platform.security.security_metrics import (
    SecurityMetrics,
)


def test_security_metrics_start_empty() -> None:
    metrics = SecurityMetrics()

    snapshot = metrics.snapshot()

    assert snapshot.authentication_failures == 0
    assert snapshot.authorization_failures == 0
    assert snapshot.failure_counts == {}
    assert snapshot.total_security_failures == 0


def test_security_metrics_record_authentication_failure() -> None:
    metrics = SecurityMetrics()

    metrics.record_authentication_failure(failure_code="authentication_required")

    snapshot = metrics.snapshot()

    assert snapshot.authentication_failures == 1
    assert snapshot.authorization_failures == 0

    assert snapshot.failure_counts == {
        "authentication_required": 1,
    }

    assert snapshot.total_security_failures == 1


def test_security_metrics_record_authorization_failure() -> None:
    metrics = SecurityMetrics()

    metrics.record_authorization_failure(failure_code="authorization_denied")

    snapshot = metrics.snapshot()

    assert snapshot.authentication_failures == 0
    assert snapshot.authorization_failures == 1

    assert snapshot.failure_counts == {
        "authorization_denied": 1,
    }


def test_security_metrics_aggregate_failure_codes() -> None:
    metrics = SecurityMetrics()

    metrics.record_authentication_failure(failure_code="authentication_required")

    metrics.record_authentication_failure(failure_code="authentication_required")

    metrics.record_authorization_failure(failure_code="authorization_denied")

    snapshot = metrics.snapshot()

    assert snapshot.authentication_failures == 2
    assert snapshot.authorization_failures == 1

    assert snapshot.failure_counts == {
        "authentication_required": 2,
        "authorization_denied": 1,
    }

    assert snapshot.total_security_failures == 3


def test_security_metrics_snapshot_to_dict() -> None:
    metrics = SecurityMetrics()

    metrics.record_authentication_failure(failure_code="authentication_required")

    metrics.record_authorization_failure(failure_code="authorization_denied")

    assert metrics.snapshot().to_dict() == {
        "authentication_failures": 1,
        "authorization_failures": 1,
        "failure_counts": {
            "authentication_required": 1,
            "authorization_denied": 1,
        },
        "total_security_failures": 2,
    }


def test_security_metrics_snapshot_is_point_in_time() -> None:
    metrics = SecurityMetrics()

    first = metrics.snapshot()

    metrics.record_authorization_failure(failure_code="authorization_denied")

    second = metrics.snapshot()

    assert first.total_security_failures == 0
    assert second.total_security_failures == 1
