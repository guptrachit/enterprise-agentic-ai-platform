from unittest.mock import Mock

from agent_platform.llm.base import LLMClient
from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.governed_runtime_factory import (
    create_governed_runtime,
)
from agent_platform.llm.model_definition import ModelDefinition
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
from agent_platform.llm.runtime_governance_config import (
    RuntimeGovernanceConfig,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
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


def create_active_policy() -> GovernedRoutingPolicy:
    return GovernedRoutingPolicy(
        versioned_policy=VersionedRoutingPolicy(
            policy=ModelPolicy(
                assignments={
                    LLMWorkload.GENERAL: "primary",
                }
            ),
            metadata=RoutingPolicyMetadata(
                name="production-routing-policy",
                version=RoutingPolicyVersion.parse("1.0.0"),
            ),
        ),
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )


def test_factory_wires_runtime_components() -> None:
    registry = RoutingPolicyRegistry()
    registry.register(create_active_policy())

    primary = create_model(
        name="primary",
    )

    client_factory = Mock(spec=lambda model: LLMClient)

    runtime = create_governed_runtime(
        registry=registry,
        models={
            "primary": primary,
        },
        client_factory=client_factory,
        config=RuntimeGovernanceConfig(),
    )

    assert runtime.registry is registry

    assert runtime.resolver.registry is registry

    assert runtime.router_factory.resolver is runtime.resolver

    assert runtime.execution_service_factory.router_factory is runtime.router_factory

    assert runtime.execution_service_factory.client_factory is client_factory

    assert runtime.refresh_service.factory is runtime.execution_service_factory

    assert runtime.refresh_service.policy_name == "production-routing-policy"


def test_factory_enables_refresh_metrics_by_default() -> None:
    registry = RoutingPolicyRegistry()
    registry.register(create_active_policy())

    runtime = create_governed_runtime(
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
        config=RuntimeGovernanceConfig(),
    )

    assert runtime.refresh_metrics is not None

    assert runtime.refresh_service.refresh_metrics is runtime.refresh_metrics


def test_factory_can_disable_refresh_metrics() -> None:
    registry = RoutingPolicyRegistry()
    registry.register(create_active_policy())

    runtime = create_governed_runtime(
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
        config=RuntimeGovernanceConfig(
            enable_refresh_metrics=False,
        ),
    )

    assert runtime.refresh_metrics is None
    assert runtime.refresh_service.refresh_metrics is None


def test_factory_preserves_refresh_configuration() -> None:
    registry = RoutingPolicyRegistry()
    registry.register(create_active_policy())

    config = RuntimeGovernanceConfig.ttl(
        ttl_seconds=120.0,
    )

    runtime = create_governed_runtime(
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
        config=config,
    )

    assert runtime.refresh_service.refresh_config.mode is RuntimePolicyRefreshMode.TTL

    assert runtime.refresh_service.refresh_config.ttl_seconds == 120.0


def test_factory_process_static_resolves_active_policy() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_active_policy()

    registry.register(policy)

    runtime = create_governed_runtime(
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
        config=RuntimeGovernanceConfig.process_static(),
    )

    assert runtime.refresh_service.policy_identifier == (
        "production-routing-policy@1.0.0"
    )


def test_factory_per_request_is_lazy() -> None:
    registry = RoutingPolicyRegistry()

    runtime = create_governed_runtime(
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
        config=RuntimeGovernanceConfig.per_request(),
    )

    assert runtime.refresh_service.policy_identifier is None

    assert (
        runtime.refresh_service.refresh_config.mode
        is RuntimePolicyRefreshMode.PER_REQUEST
    )
