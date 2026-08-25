import pytest

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
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
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_governed_policy(
    *,
    name: str = "production-routing-policy",
    version: str = "1.0.0",
    status: RoutingPolicyLifecycleStatus = (RoutingPolicyLifecycleStatus.ACTIVE),
) -> GovernedRoutingPolicy:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: (
                "primary",
                "backup",
            ),
        }
    )

    versioned_policy = VersionedRoutingPolicy(
        policy=policy,
        metadata=RoutingPolicyMetadata(
            name=name,
            version=RoutingPolicyVersion.parse(version),
        ),
    )

    return GovernedRoutingPolicy(
        versioned_policy=versioned_policy,
        status=status,
    )


def test_resolver_returns_active_versioned_policy() -> None:
    registry = RoutingPolicyRegistry()

    governed = create_governed_policy(
        version="1.5.0",
    )

    registry.register(governed)

    resolver = ActiveRoutingPolicyResolver(registry)

    result = resolver.resolve("production-routing-policy")

    assert result is governed.versioned_policy

    assert result.identifier == ("production-routing-policy@1.5.0")


def test_resolver_ignores_non_active_versions() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_governed_policy(
        version="1.6.0",
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    active = create_governed_policy(
        version="1.5.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry.register(candidate)

    registry.register(active)

    resolver = ActiveRoutingPolicyResolver(registry)

    result = resolver.resolve("production-routing-policy")

    assert result.identifier == ("production-routing-policy@1.5.0")


def test_resolver_raises_when_no_active_policy_exists() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_governed_policy(
            version="1.6.0",
            status=RoutingPolicyLifecycleStatus.CANDIDATE,
        )
    )

    resolver = ActiveRoutingPolicyResolver(registry)

    with pytest.raises(
        LookupError,
        match="No active routing policy found",
    ):
        resolver.resolve("production-routing-policy")


def test_resolver_raises_for_missing_policy_name() -> None:
    registry = RoutingPolicyRegistry()

    resolver = ActiveRoutingPolicyResolver(registry)

    with pytest.raises(
        LookupError,
        match="missing-policy",
    ):
        resolver.resolve("missing-policy")


def test_resolver_supports_multiple_policy_names() -> None:
    registry = RoutingPolicyRegistry()

    production = create_governed_policy(
        name="production-policy",
        version="2.0.0",
    )

    experimental = create_governed_policy(
        name="experimental-policy",
        version="3.0.0",
    )

    registry.register(production)

    registry.register(experimental)

    resolver = ActiveRoutingPolicyResolver(registry)

    assert resolver.resolve("production-policy").identifier == (
        "production-policy@2.0.0"
    )

    assert resolver.resolve("experimental-policy").identifier == (
        "experimental-policy@3.0.0"
    )


def test_resolver_reflects_registry_activation_changes() -> None:
    registry = RoutingPolicyRegistry()

    old = create_governed_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    new = create_governed_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(old)

    registry.register(new)

    resolver = ActiveRoutingPolicyResolver(registry)

    before = resolver.resolve("production-routing-policy")

    assert before.identifier == ("production-routing-policy@1.0.0")

    registry.activate(new.identifier)

    after = resolver.resolve("production-routing-policy")

    assert after.identifier == ("production-routing-policy@1.1.0")
