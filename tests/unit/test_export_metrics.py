import pytest

from agent_platform.observability.export_metrics import (
    ExportMetrics,
)


def test_export_metrics_start_empty() -> None:
    metrics = ExportMetrics()

    snapshot = metrics.snapshot()

    assert snapshot.total_exports == 0
    assert snapshot.successful_exports == 0
    assert snapshot.failed_exports == 0
    assert snapshot.success_rate == 0.0
    assert snapshot.failure_rate == 0.0


def test_export_metrics_record_success() -> None:
    metrics = ExportMetrics()

    metrics.record_success()

    snapshot = metrics.snapshot()

    assert snapshot.total_exports == 1
    assert snapshot.successful_exports == 1
    assert snapshot.failed_exports == 0
    assert snapshot.success_rate == 1.0
    assert snapshot.failure_rate == 0.0


def test_export_metrics_record_failure() -> None:
    metrics = ExportMetrics()

    metrics.record_failure()

    snapshot = metrics.snapshot()

    assert snapshot.total_exports == 1
    assert snapshot.successful_exports == 0
    assert snapshot.failed_exports == 1
    assert snapshot.failure_rate == 1.0


def test_export_metrics_calculate_rates() -> None:
    metrics = ExportMetrics()

    metrics.record_success()
    metrics.record_success()
    metrics.record_failure()

    snapshot = metrics.snapshot()

    assert snapshot.total_exports == 3
    assert snapshot.successful_exports == 2
    assert snapshot.failed_exports == 1

    assert snapshot.success_rate == pytest.approx(2 / 3)

    assert snapshot.failure_rate == pytest.approx(1 / 3)


def test_export_metrics_to_dict() -> None:
    metrics = ExportMetrics()

    metrics.record_success()
    metrics.record_failure()

    assert metrics.snapshot().to_dict() == {
        "total_exports": 2,
        "successful_exports": 1,
        "failed_exports": 1,
        "success_rate": 0.5,
        "failure_rate": 0.5,
    }
