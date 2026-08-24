import json

import pytest

from agent_platform.llm.routing_metrics import (
    RoutingMetrics,
    RoutingMetricsSnapshot,
)


def test_routing_metrics_defaults() -> None:
    metrics = RoutingMetrics()

    assert metrics.total_requests == 0
    assert metrics.successful_requests == 0
    assert metrics.failed_requests == 0
    assert metrics.fallback_requests == 0
    assert metrics.model_selection_counts == {}
    assert metrics.executed_model_counts == {}
    assert metrics.rejection_reason_counts == {}
    assert metrics.workload_counts == {}

    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 0.0
    assert metrics.fallback_rate == 0.0


def test_routing_metrics_records_successful_request() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="classification",
        selected_model="classification_primary",
        executed_model="classification_primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    assert metrics.total_requests == 1
    assert metrics.successful_requests == 1
    assert metrics.failed_requests == 0
    assert metrics.fallback_requests == 0

    assert metrics.model_selection_counts == {
        "classification_primary": 1,
    }

    assert metrics.executed_model_counts == {
        "classification_primary": 1,
    }

    assert metrics.rejection_reason_counts == {
        "selected": 1,
    }

    assert metrics.workload_counts == {
        "classification": 1,
    }

    assert metrics.success_rate == 1.0
    assert metrics.failure_rate == 0.0
    assert metrics.fallback_rate == 0.0


def test_routing_metrics_records_fallback() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=True,
    )

    assert metrics.total_requests == 1
    assert metrics.successful_requests == 1
    assert metrics.fallback_requests == 1

    assert metrics.model_selection_counts == {
        "primary": 1,
    }

    assert metrics.executed_model_counts == {
        "backup": 1,
    }

    assert metrics.success_rate == 1.0
    assert metrics.failure_rate == 0.0
    assert metrics.fallback_rate == 1.0


def test_routing_metrics_records_failed_request() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="reasoning",
        selected_model="reasoning_primary",
        executed_model="reasoning_primary",
        routing_reason_codes=("selected",),
        success=False,
        fallback_used=False,
    )

    assert metrics.total_requests == 1
    assert metrics.successful_requests == 0
    assert metrics.failed_requests == 1

    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 1.0
    assert metrics.fallback_rate == 0.0


def test_routing_metrics_accumulates_multiple_requests() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="classification",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=(
            "constraint_rejected",
            "selected",
        ),
        success=True,
        fallback_used=False,
    )

    metrics.record_request(
        workload="classification",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=(
            "constraint_rejected",
            "selected",
        ),
        success=True,
        fallback_used=True,
    )

    metrics.record_request(
        workload="reasoning",
        selected_model="reasoning_primary",
        executed_model="reasoning_primary",
        routing_reason_codes=("selected",),
        success=False,
        fallback_used=False,
    )

    assert metrics.total_requests == 3
    assert metrics.successful_requests == 2
    assert metrics.failed_requests == 1
    assert metrics.fallback_requests == 1

    assert metrics.success_rate == pytest.approx(2 / 3)

    assert metrics.failure_rate == pytest.approx(1 / 3)

    assert metrics.fallback_rate == pytest.approx(1 / 3)


def test_routing_metrics_handles_missing_executed_model() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model=None,
        routing_reason_codes=("selected",),
        success=False,
        fallback_used=False,
    )

    assert metrics.total_requests == 1
    assert metrics.failed_requests == 1
    assert metrics.executed_model_counts == {}


def test_routing_metric_rates_change_as_requests_accumulate() -> None:
    metrics = RoutingMetrics()

    for success, fallback_used in (
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ):
        metrics.record_request(
            workload="general",
            selected_model="primary",
            executed_model="primary",
            routing_reason_codes=("selected",),
            success=success,
            fallback_used=fallback_used,
        )

    assert metrics.total_requests == 4
    assert metrics.success_rate == 0.5
    assert metrics.failure_rate == 0.5
    assert metrics.fallback_rate == 0.5


def test_routing_metrics_snapshot_contains_current_values() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="classification",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=(
            "constraint_rejected",
            "selected",
        ),
        success=True,
        fallback_used=True,
    )

    snapshot = metrics.snapshot()

    assert isinstance(
        snapshot,
        RoutingMetricsSnapshot,
    )

    assert snapshot.total_requests == 1
    assert snapshot.successful_requests == 1
    assert snapshot.failed_requests == 0
    assert snapshot.fallback_requests == 1

    assert snapshot.success_rate == 1.0
    assert snapshot.failure_rate == 0.0
    assert snapshot.fallback_rate == 1.0

    assert snapshot.model_selection_counts == {
        "primary": 1,
    }

    assert snapshot.executed_model_counts == {
        "backup": 1,
    }

    assert snapshot.rejection_reason_counts == {
        "constraint_rejected": 1,
        "selected": 1,
    }

    assert snapshot.workload_counts == {
        "classification": 1,
    }


def test_routing_metrics_snapshot_is_point_in_time() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    snapshot = metrics.snapshot()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=True,
    )

    assert snapshot.total_requests == 1

    assert snapshot.model_selection_counts == {
        "primary": 1,
    }

    assert snapshot.executed_model_counts == {
        "primary": 1,
    }

    assert metrics.total_requests == 2


def test_routing_metrics_snapshot_mappings_are_immutable() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    snapshot = metrics.snapshot()

    with pytest.raises(TypeError):
        snapshot.model_selection_counts["primary"] = 99  # type: ignore[index]

    with pytest.raises(TypeError):
        snapshot.executed_model_counts["backup"] = 1  # type: ignore[index]

    assert metrics.model_selection_counts == {
        "primary": 1,
    }


def test_routing_metrics_snapshot_to_dict() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="classification",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=(
            "constraint_rejected",
            "selected",
        ),
        success=True,
        fallback_used=True,
    )

    snapshot = metrics.snapshot()

    payload = snapshot.to_dict()

    assert payload == {
        "total_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "fallback_requests": 1,
        "metrics_export_failures": 0,
        "success_rate": 1.0,
        "failure_rate": 0.0,
        "fallback_rate": 1.0,
        "model_selection_counts": {
            "primary": 1,
        },
        "executed_model_counts": {
            "backup": 1,
        },
        "rejection_reason_counts": {
            "constraint_rejected": 1,
            "selected": 1,
        },
        "workload_counts": {
            "classification": 1,
        },
    }


def test_routing_metrics_snapshot_to_dict_is_independent() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="primary",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=False,
    )

    snapshot = metrics.snapshot()
    payload = snapshot.to_dict()

    model_counts = payload["model_selection_counts"]

    assert isinstance(
        model_counts,
        dict,
    )

    model_counts["primary"] = 999

    assert snapshot.model_selection_counts == {
        "primary": 1,
    }

    assert metrics.model_selection_counts == {
        "primary": 1,
    }


def test_routing_metrics_snapshot_is_json_serializable() -> None:
    metrics = RoutingMetrics()

    metrics.record_request(
        workload="general",
        selected_model="primary",
        executed_model="backup",
        routing_reason_codes=("selected",),
        success=True,
        fallback_used=True,
    )

    payload = metrics.snapshot().to_dict()

    serialized = json.dumps(
        payload,
        sort_keys=True,
    )

    decoded = json.loads(serialized)

    assert decoded["total_requests"] == 1

    assert decoded["model_selection_counts"] == {
        "primary": 1,
    }

    assert decoded["executed_model_counts"] == {
        "backup": 1,
    }


def test_routing_metrics_records_export_failure() -> None:
    metrics = RoutingMetrics()

    metrics.record_export_failure()

    assert metrics.metrics_export_failures == 1

    metrics.record_export_failure()

    assert metrics.metrics_export_failures == 2


def test_routing_metrics_snapshot_contains_export_failures() -> None:
    metrics = RoutingMetrics()

    metrics.record_export_failure()
    metrics.record_export_failure()

    snapshot = metrics.snapshot()

    assert snapshot.metrics_export_failures == 2

    payload = snapshot.to_dict()

    assert payload["metrics_export_failures"] == 2


def test_routing_metrics_records_export_failure() -> None:
    metrics = RoutingMetrics()

    metrics.record_export_failure()

    assert metrics.metrics_export_failures == 1

    metrics.record_export_failure()

    assert metrics.metrics_export_failures == 2


def test_routing_metrics_snapshot_contains_export_failures() -> None:
    metrics = RoutingMetrics()

    metrics.record_export_failure()
    metrics.record_export_failure()

    snapshot = metrics.snapshot()

    assert snapshot.metrics_export_failures == 2

    payload = snapshot.to_dict()

    assert payload["metrics_export_failures"] == 2
