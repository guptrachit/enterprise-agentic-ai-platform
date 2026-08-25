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
from agent_platform.llm.routing_policy_registry_validator import (
    validate_routing_policy_registry,
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
) -> GovernedRoutingPolicy:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "primary",
        }
    )

    return GovernedRoutingPolicy(
        versioned_policy=VersionedRoutingPolicy(
            policy=policy,
            metadata=RoutingPolicyMetadata(
                name=name,
                version=RoutingPolicyVersion.parse(version),
            ),
        ),
        status=status,
    )


def test_empty_registry_is_valid() -> None:
    registry = RoutingPolicyRegistry()

    result = validate_routing_policy_registry(registry)

    assert result.valid is True
    assert result.errors == ()
    assert result.error_count == 0


def test_registered_draft_policy_is_valid() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(create_policy())

    result = validate_routing_policy_registry(registry)

    assert result.valid is True


def test_single_active_policy_is_valid() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
        )
    )

    registry.register(
        create_policy(
            version="1.1.0",
            status=RoutingPolicyLifecycleStatus.APPROVED,
        )
    )

    result = validate_routing_policy_registry(registry)

    assert result.valid is True
    assert result.error_count == 0


def test_activation_history_is_valid_after_activation() -> None:
    registry = RoutingPolicyRegistry()

    current = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(current)
    registry.register(candidate)

    registry.activate(candidate.identifier)

    result = validate_routing_policy_registry(registry)

    assert result.valid is True
    assert result.errors == ()


def test_multiple_policy_names_are_valid() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            name="production-policy",
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
        )
    )

    registry.register(
        create_policy(
            name="experimental-policy",
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
        )
    )

    result = validate_routing_policy_registry(registry)

    assert result.valid is True


def test_validation_detects_multiple_active_versions_if_state_is_corrupted() -> None:
    registry = RoutingPolicyRegistry()

    first = create_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    second = create_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry._policies[first.identifier] = first
    registry._policies[second.identifier] = second

    result = validate_routing_policy_registry(registry)

    assert result.valid is False
    assert result.error_count == 1

    assert result.errors == (
        "Routing policy 'production-policy' has multiple active versions.",
    )


def test_validation_detects_unregistered_activated_policy_in_history() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(candidate)

    registry.activate(candidate.identifier)

    registry._policies.pop(candidate.identifier)

    result = validate_routing_policy_registry(registry)

    assert result.valid is False

    assert any("unregistered activated policy" in error for error in result.errors)


def test_validation_detects_unregistered_retired_policy_in_history() -> None:
    registry = RoutingPolicyRegistry()

    old = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(old)
    registry.register(candidate)

    registry.activate(candidate.identifier)

    registry._policies.pop(old.identifier)

    result = validate_routing_policy_registry(registry)

    assert result.valid is False

    assert any("unregistered retired policy" in error for error in result.errors)


def test_validation_detects_inconsistent_activated_policy_status() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(candidate)

    registry.activate(candidate.identifier)

    registry._policies[candidate.identifier] = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    result = validate_routing_policy_registry(registry)

    assert result.valid is False

    assert any("inconsistent lifecycle status" in error for error in result.errors)


def test_validation_result_counts_multiple_errors() -> None:
    registry = RoutingPolicyRegistry()

    first = create_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    second = create_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry._policies[first.identifier] = first
    registry._policies[second.identifier] = second

    result = validate_routing_policy_registry(registry)

    assert result.valid is False
    assert result.error_count >= 1
