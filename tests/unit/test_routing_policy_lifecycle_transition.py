import pytest

from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.routing_policy_lifecycle_transition import (
    RoutingPolicyLifecycleTransitionStatus,
    evaluate_lifecycle_transition,
)


@pytest.mark.parametrize(
    (
        "current_status",
        "target_status",
    ),
    (
        (
            RoutingPolicyLifecycleStatus.DRAFT,
            RoutingPolicyLifecycleStatus.CANDIDATE,
        ),
        (
            RoutingPolicyLifecycleStatus.CANDIDATE,
            RoutingPolicyLifecycleStatus.APPROVED,
        ),
        (
            RoutingPolicyLifecycleStatus.APPROVED,
            RoutingPolicyLifecycleStatus.ACTIVE,
        ),
        (
            RoutingPolicyLifecycleStatus.ACTIVE,
            RoutingPolicyLifecycleStatus.RETIRED,
        ),
    ),
)
def test_valid_lifecycle_transitions_are_allowed(
    current_status: RoutingPolicyLifecycleStatus,
    target_status: RoutingPolicyLifecycleStatus,
) -> None:
    result = evaluate_lifecycle_transition(
        current_status=current_status,
        target_status=target_status,
    )

    assert result.status is RoutingPolicyLifecycleTransitionStatus.ALLOWED

    assert result.allowed is True
    assert result.current_status is current_status
    assert result.target_status is target_status


@pytest.mark.parametrize(
    (
        "current_status",
        "target_status",
    ),
    (
        (
            RoutingPolicyLifecycleStatus.DRAFT,
            RoutingPolicyLifecycleStatus.ACTIVE,
        ),
        (
            RoutingPolicyLifecycleStatus.DRAFT,
            RoutingPolicyLifecycleStatus.APPROVED,
        ),
        (
            RoutingPolicyLifecycleStatus.CANDIDATE,
            RoutingPolicyLifecycleStatus.ACTIVE,
        ),
        (
            RoutingPolicyLifecycleStatus.APPROVED,
            RoutingPolicyLifecycleStatus.RETIRED,
        ),
        (
            RoutingPolicyLifecycleStatus.ACTIVE,
            RoutingPolicyLifecycleStatus.CANDIDATE,
        ),
        (
            RoutingPolicyLifecycleStatus.RETIRED,
            RoutingPolicyLifecycleStatus.ACTIVE,
        ),
    ),
)
def test_invalid_lifecycle_transitions_are_rejected(
    current_status: RoutingPolicyLifecycleStatus,
    target_status: RoutingPolicyLifecycleStatus,
) -> None:
    result = evaluate_lifecycle_transition(
        current_status=current_status,
        target_status=target_status,
    )

    assert result.status is RoutingPolicyLifecycleTransitionStatus.REJECTED

    assert result.allowed is False


@pytest.mark.parametrize(
    "status",
    tuple(RoutingPolicyLifecycleStatus),
)
def test_same_status_transition_is_rejected(
    status: RoutingPolicyLifecycleStatus,
) -> None:
    result = evaluate_lifecycle_transition(
        current_status=status,
        target_status=status,
    )

    assert result.allowed is False

    assert result.status is RoutingPolicyLifecycleTransitionStatus.REJECTED


def test_transition_reason_is_human_readable() -> None:
    result = evaluate_lifecycle_transition(
        current_status=RoutingPolicyLifecycleStatus.DRAFT,
        target_status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    assert result.reason == (
        "Lifecycle transition from 'draft' to 'candidate' is allowed."
    )


def test_rejected_transition_reason_is_human_readable() -> None:
    result = evaluate_lifecycle_transition(
        current_status=RoutingPolicyLifecycleStatus.DRAFT,
        target_status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    assert result.reason == (
        "Lifecycle transition from 'draft' to 'active' is not allowed."
    )
