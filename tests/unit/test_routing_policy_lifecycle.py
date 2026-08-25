from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)


def test_lifecycle_status_values() -> None:
    assert RoutingPolicyLifecycleStatus.DRAFT.value == "draft"
    assert RoutingPolicyLifecycleStatus.CANDIDATE.value == "candidate"
    assert RoutingPolicyLifecycleStatus.APPROVED.value == "approved"
    assert RoutingPolicyLifecycleStatus.ACTIVE.value == "active"
    assert RoutingPolicyLifecycleStatus.RETIRED.value == "retired"


def test_lifecycle_status_is_string_compatible() -> None:
    assert RoutingPolicyLifecycleStatus.DRAFT == "draft"
    assert RoutingPolicyLifecycleStatus.ACTIVE == "active"


def test_lifecycle_status_can_be_created_from_string() -> None:
    assert (
        RoutingPolicyLifecycleStatus("candidate")
        is RoutingPolicyLifecycleStatus.CANDIDATE
    )

    assert (
        RoutingPolicyLifecycleStatus("retired") is RoutingPolicyLifecycleStatus.RETIRED
    )


def test_lifecycle_contains_expected_statuses() -> None:
    assert tuple(status.value for status in RoutingPolicyLifecycleStatus) == (
        "draft",
        "candidate",
        "approved",
        "active",
        "retired",
    )
