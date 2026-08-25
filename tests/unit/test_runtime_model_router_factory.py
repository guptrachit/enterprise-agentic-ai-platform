import pytest

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_router import ModelRouter
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
from agent_platform.llm.runtime_model_router_factory import (
    RuntimeModelRouterFactory,
    RuntimeModelRouterResolution,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
    workload: LLMWorkload = LLMWorkload.GENERAL,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider="openai",
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                workload,
            }
        ),
    )


def create_governed_policy(
    *,
    version: str,
    status: RoutingPolicyLifecycleStatus,
    assignments: dict[
        LLMWorkload,
        str | tuple[str, ...],
    ],
    name: str = "production-routing-policy",
) -> GovernedRoutingPolicy:
    policy = ModelPolicy(assignments=assignments)

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


def test_factory_creates_model_router_from_active_policy() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    registry = RoutingPolicyRegistry()

    active = create_governed_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        assignments={
            LLMWorkload.GENERAL: (
                "primary",
                "backup",
            ),
        },
    )

    registry.register(active)

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "primary": primary,
            "backup": backup,
        },
    )

    router = factory.create("production-routing-policy")

    assert isinstance(
        router,
        ModelRouter,
    )

    decision = router.route_decision(LLMWorkload.GENERAL)

    assert decision.selected_model is primary


def test_factory_uses_active_policy_assignments() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    registry = RoutingPolicyRegistry()

    active = create_governed_policy(
        version="2.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        assignments={
            LLMWorkload.GENERAL: (
                "backup",
                "primary",
            ),
        },
    )

    registry.register(active)

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "primary": primary,
            "backup": backup,
        },
    )

    router = factory.create("production-routing-policy")

    decision = router.route_decision(LLMWorkload.GENERAL)

    assert decision.selected_model is backup


def test_factory_reflects_policy_activation_change() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    registry = RoutingPolicyRegistry()

    old = create_governed_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        assignments={
            LLMWorkload.GENERAL: "primary",
        },
    )

    new = create_governed_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
        assignments={
            LLMWorkload.GENERAL: "backup",
        },
    )

    registry.register(old)
    registry.register(new)

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "primary": primary,
            "backup": backup,
        },
    )

    before = factory.resolve("production-routing-policy")

    assert before.policy_identifier == ("production-routing-policy@1.0.0")

    assert before.router.route_decision(LLMWorkload.GENERAL).selected_model is primary

    registry.activate(new.identifier)

    after = factory.resolve("production-routing-policy")

    assert after.policy_identifier == ("production-routing-policy@1.1.0")

    assert after.router.route_decision(LLMWorkload.GENERAL).selected_model is backup


def test_factory_raises_when_no_active_policy_exists() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_governed_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.CANDIDATE,
            assignments={
                LLMWorkload.GENERAL: "primary",
            },
        )
    )

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "primary": primary,
        },
    )

    with pytest.raises(
        LookupError,
        match="No active routing policy found",
    ):
        factory.resolve("production-routing-policy")


def test_factory_supports_multiple_policy_names() -> None:
    production_model = create_model(
        name="production_model",
    )

    experimental_model = create_model(
        name="experimental_model",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_governed_policy(
            name="production-policy",
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            assignments={
                LLMWorkload.GENERAL: "production_model",
            },
        )
    )

    registry.register(
        create_governed_policy(
            name="experimental-policy",
            version="3.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            assignments={
                LLMWorkload.GENERAL: "experimental_model",
            },
        )
    )

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "production_model": production_model,
            "experimental_model": experimental_model,
        },
    )

    production = factory.resolve("production-policy")

    experimental = factory.resolve("experimental-policy")

    assert production.policy_identifier == ("production-policy@1.0.0")

    assert experimental.policy_identifier == ("experimental-policy@3.0.0")

    assert (
        production.router.route_decision(LLMWorkload.GENERAL).selected_model
        is production_model
    )

    assert (
        experimental.router.route_decision(LLMWorkload.GENERAL).selected_model
        is experimental_model
    )


def test_factory_copies_model_mapping() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_governed_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            assignments={
                LLMWorkload.GENERAL: "primary",
            },
        )
    )

    models = {
        "primary": primary,
    }

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models=models,
    )

    models.clear()

    resolution = factory.resolve("production-routing-policy")

    assert (
        resolution.router.route_decision(LLMWorkload.GENERAL).selected_model is primary
    )


def test_resolve_returns_policy_identifier() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_governed_policy(
            version="4.2.1",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            assignments={
                LLMWorkload.GENERAL: "primary",
            },
        )
    )

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "primary": primary,
        },
    )

    resolution = factory.resolve("production-routing-policy")

    assert isinstance(
        resolution,
        RuntimeModelRouterResolution,
    )

    assert resolution.policy_identifier == ("production-routing-policy@4.2.1")

    assert isinstance(
        resolution.router,
        ModelRouter,
    )


def test_create_remains_backward_compatible() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_governed_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            assignments={
                LLMWorkload.GENERAL: "primary",
            },
        )
    )

    factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "primary": primary,
        },
    )

    router = factory.create("production-routing-policy")

    assert isinstance(
        router,
        ModelRouter,
    )

    assert router.route_decision(LLMWorkload.GENERAL).selected_model is primary
