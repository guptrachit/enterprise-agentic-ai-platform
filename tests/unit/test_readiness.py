from unittest.mock import Mock

from agent_platform.readiness import (
    ReadinessStatus,
    assess_runtime_readiness,
)


def create_runtime(
    *,
    resolved: bool,
):
    runtime = Mock()

    snapshot = Mock()

    snapshot.to_dict.return_value = {
        "resolved": resolved,
    }

    runtime.health_snapshot.return_value = snapshot

    return runtime


def test_runtime_is_ready_when_resolved() -> None:
    assessment = assess_runtime_readiness(create_runtime(resolved=True))

    assert assessment.status is ReadinessStatus.READY
    assert assessment.reasons == ()


def test_runtime_is_not_ready_when_unresolved() -> None:
    assessment = assess_runtime_readiness(create_runtime(resolved=False))

    assert assessment.status is ReadinessStatus.NOT_READY

    assert assessment.reasons == ("routing_policy_unresolved",)


def test_readiness_assessment_to_dict() -> None:
    assessment = assess_runtime_readiness(create_runtime(resolved=True))

    assert assessment.to_dict() == {
        "status": "ready",
        "reasons": [],
    }
