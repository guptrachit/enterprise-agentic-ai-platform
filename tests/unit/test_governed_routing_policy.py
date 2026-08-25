from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_versioned_policy() -> VersionedRoutingPolicy:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: (
                "primary",
                "backup",
            ),
        }
    )

    metadata = RoutingPolicyMetadata(
        name="production-routing-policy",
        version=RoutingPolicyVersion(
            major=1,
            minor=5,
            patch=0,
        ),
        description="Production routing policy",
    )

    return VersionedRoutingPolicy(
        policy=policy,
        metadata=metadata,
    )


def test_governed_policy_preserves_versioned_policy() -> None:
    versioned_policy = create_versioned_policy()

    governed = GovernedRoutingPolicy(
        versioned_policy=versioned_policy,
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    assert governed.versioned_policy is versioned_policy

    assert governed.status is RoutingPolicyLifecycleStatus.CANDIDATE


def test_governed_policy_exposes_identifier() -> None:
    governed = GovernedRoutingPolicy(
        versioned_policy=create_versioned_policy(),
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    assert governed.identifier == ("production-routing-policy@1.5.0")


def test_governed_policy_exposes_name_and_version() -> None:
    governed = GovernedRoutingPolicy(
        versioned_policy=create_versioned_policy(),
        status=RoutingPolicyLifecycleStatus.DRAFT,
    )

    assert governed.policy_name == ("production-routing-policy")

    assert governed.version == "1.5.0"


def test_active_policy_reports_active() -> None:
    governed = GovernedRoutingPolicy(
        versioned_policy=create_versioned_policy(),
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    assert governed.is_active is True
    assert governed.is_retired is False


def test_retired_policy_reports_retired() -> None:
    governed = GovernedRoutingPolicy(
        versioned_policy=create_versioned_policy(),
        status=RoutingPolicyLifecycleStatus.RETIRED,
    )

    assert governed.is_active is False
    assert governed.is_retired is True


def test_non_active_non_retired_policy_flags_are_false() -> None:
    governed = GovernedRoutingPolicy(
        versioned_policy=create_versioned_policy(),
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    assert governed.is_active is False
    assert governed.is_retired is False
