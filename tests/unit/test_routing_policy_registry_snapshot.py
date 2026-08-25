import json

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
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)
from agent_platform.llm.routing_policy_registry_snapshot import (
    RoutingPolicyRegistrySnapshot,
    create_registry_snapshot,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_policy(
    *,
    name: str = "production-policy",
    version: str = "1.0.0",
    status: RoutingPolicyLifecycleStatus = (RoutingPolicyLifecycleStatus.DRAFT),
    description: str | None = None,
) -> GovernedRoutingPolicy:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "primary",
        }
    )

    versioned_policy = VersionedRoutingPolicy(
        policy=policy,
        metadata=RoutingPolicyMetadata(
            name=name,
            version=RoutingPolicyVersion.parse(version),
            description=description,
        ),
    )

    return GovernedRoutingPolicy(
        versioned_policy=versioned_policy,
        status=status,
    )


def test_empty_registry_snapshot() -> None:
    registry = RoutingPolicyRegistry()

    snapshot = create_registry_snapshot(registry)

    assert isinstance(
        snapshot,
        RoutingPolicyRegistrySnapshot,
    )

    assert snapshot.policy_count == 0
    assert snapshot.activation_count == 0
    assert snapshot.policies == ()
    assert snapshot.activation_history == ()


def test_registry_snapshot_contains_registered_policies() -> None:
    registry = RoutingPolicyRegistry()

    first = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.RETIRED,
        description="Original policy",
    )

    second = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
        description="Candidate policy",
    )

    registry.register(first)
    registry.register(second)

    snapshot = create_registry_snapshot(registry)

    assert snapshot.policy_count == 2

    assert snapshot.policies[0].identifier == ("production-policy@1.0.0")

    assert snapshot.policies[0].status == "retired"

    assert snapshot.policies[0].description == ("Original policy")

    assert snapshot.policies[1].identifier == ("production-policy@1.1.0")

    assert snapshot.policies[1].status == "candidate"


def test_registry_snapshot_contains_activation_history() -> None:
    registry = RoutingPolicyRegistry()

    first = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    second = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(first)
    registry.register(second)

    registry.activate(second.identifier)

    snapshot = create_registry_snapshot(registry)

    assert snapshot.activation_count == 1

    event = snapshot.activation_history[0]

    assert event.policy_name == ("production-policy")

    assert event.activated_policy_identifier == ("production-policy@1.1.0")

    assert event.activated_version == "1.1.0"

    assert event.retired_policy_identifier == ("production-policy@1.0.0")

    assert event.retired_version == "1.0.0"


def test_registry_snapshot_is_point_in_time() -> None:
    registry = RoutingPolicyRegistry()

    first = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(first)

    snapshot = create_registry_snapshot(registry)

    second = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.DRAFT,
    )

    registry.register(second)

    registry.activate(first.identifier)

    assert snapshot.policy_count == 1
    assert snapshot.activation_count == 0

    assert len(registry.list_all()) == 2

    assert len(registry.activation_history()) == 1


def test_registry_snapshot_to_dict() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_policy(
        version="2.0.0",
        status=RoutingPolicyLifecycleStatus.DRAFT,
        description="Version two policy",
    )

    registry.register(policy)

    payload = create_registry_snapshot(registry).to_dict()

    assert payload == {
        "policy_count": 1,
        "activation_count": 0,
        "policies": [
            {
                "identifier": ("production-policy@2.0.0"),
                "policy_name": "production-policy",
                "version": "2.0.0",
                "status": "draft",
                "description": ("Version two policy"),
            }
        ],
        "activation_history": [],
    }


def test_registry_snapshot_to_dict_is_independent() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
        )
    )

    snapshot = create_registry_snapshot(registry)

    payload = snapshot.to_dict()

    policies = payload["policies"]

    assert isinstance(
        policies,
        list,
    )

    policies.append(
        {
            "identifier": "mutated",
        }
    )

    assert snapshot.policy_count == 1

    assert snapshot.policies[0].identifier == ("production-policy@1.0.0")


def test_registry_snapshot_is_json_serializable() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.APPROVED,
        )
    )

    snapshot = create_registry_snapshot(registry)

    serialized = snapshot.to_json()

    payload = json.loads(serialized)

    assert payload["policy_count"] == 1

    assert payload["policies"][0]["identifier"] == ("production-policy@1.0.0")


def test_registry_snapshot_json_is_stable() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
        )
    )

    snapshot = create_registry_snapshot(registry)

    first = snapshot.to_json()
    second = snapshot.to_json()

    assert first == second
