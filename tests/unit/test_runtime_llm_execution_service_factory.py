from unittest.mock import Mock

import pytest

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_metrics import RoutingMetrics
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
from agent_platform.llm.runtime_llm_execution_service_factory import (
    RuntimeLLMExecutionServiceFactory,
)
from agent_platform.llm.runtime_model_router_factory import (
    RuntimeModelRouterFactory,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider="openai",
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )


def create_policy(
    *,
    version: str,
    status: RoutingPolicyLifecycleStatus,
    selected_model: str,
) -> GovernedRoutingPolicy:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: selected_model,
        }
    )

    return GovernedRoutingPolicy(
        versioned_policy=VersionedRoutingPolicy(
            policy=policy,
            metadata=RoutingPolicyMetadata(
                name="production-routing-policy",
                version=RoutingPolicyVersion.parse(version),
            ),
        ),
        status=status,
    )


def create_factory(
    *,
    registry: RoutingPolicyRegistry,
    models: dict[str, ModelDefinition],
    client_factory=None,
    metrics: RoutingMetrics | None = None,
    metrics_exporter=None,
) -> RuntimeLLMExecutionServiceFactory:
    if client_factory is None:
        client_factory = Mock()
    router_factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models=models,
    )

    return RuntimeLLMExecutionServiceFactory(
        router_factory=router_factory,
        client_factory=client_factory,
        metrics=metrics,
        metrics_exporter=metrics_exporter,
    )


def test_factory_creates_execution_service_with_policy_identifier() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.2.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            selected_model="primary",
        )
    )

    factory = create_factory(
        registry=registry,
        models={
            "primary": primary,
        },
    )

    service = factory.create("production-routing-policy")

    assert service.policy_identifier == ("production-routing-policy@1.2.0")


def test_factory_execution_service_uses_active_policy_router() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="2.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            selected_model="backup",
        )
    )

    factory = create_factory(
        registry=registry,
        models={
            "primary": primary,
            "backup": backup,
        },
    )

    service = factory.create("production-routing-policy")

    decision = service.router.route_decision(LLMWorkload.GENERAL)

    assert decision.selected_model is backup


def test_factory_preserves_client_factory() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            selected_model="primary",
        )
    )

    client_factory = Mock()

    factory = create_factory(
        registry=registry,
        models={
            "primary": primary,
        },
        client_factory=client_factory,
    )

    service = factory.create("production-routing-policy")

    assert service.client_factory is client_factory


def test_factory_preserves_metrics_dependencies() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            selected_model="primary",
        )
    )

    metrics = RoutingMetrics()
    exporter = Mock()

    factory = create_factory(
        registry=registry,
        models={
            "primary": primary,
        },
        metrics=metrics,
        metrics_exporter=exporter,
    )

    service = factory.create("production-routing-policy")

    assert service.metrics is metrics
    assert service.metrics_exporter is exporter


def test_factory_reflects_policy_activation_change() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    registry = RoutingPolicyRegistry()

    old = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        selected_model="primary",
    )

    new = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
        selected_model="backup",
    )

    registry.register(old)
    registry.register(new)

    factory = create_factory(
        registry=registry,
        models={
            "primary": primary,
            "backup": backup,
        },
    )

    before = factory.create("production-routing-policy")

    assert before.policy_identifier == ("production-routing-policy@1.0.0")

    assert before.router.route_decision(LLMWorkload.GENERAL).selected_model is primary

    registry.activate(new.identifier)

    after = factory.create("production-routing-policy")

    assert after.policy_identifier == ("production-routing-policy@1.1.0")

    assert after.router.route_decision(LLMWorkload.GENERAL).selected_model is backup


def test_factory_raises_when_no_active_policy_exists() -> None:
    primary = create_model(
        name="primary",
    )

    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.CANDIDATE,
            selected_model="primary",
        )
    )

    factory = create_factory(
        registry=registry,
        models={
            "primary": primary,
        },
    )

    with pytest.raises(
        LookupError,
        match="No active routing policy found",
    ):
        factory.create("production-routing-policy")
