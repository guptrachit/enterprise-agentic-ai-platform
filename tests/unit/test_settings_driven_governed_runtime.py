from unittest.mock import Mock

from agent_platform.config import Settings
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
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.settings_driven_governed_runtime import (
    create_settings_driven_governed_runtime,
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


def create_active_policy(
    *,
    name: str = "production-routing-policy",
) -> GovernedRoutingPolicy:
    return GovernedRoutingPolicy(
        versioned_policy=VersionedRoutingPolicy(
            policy=ModelPolicy(
                assignments={
                    LLMWorkload.GENERAL: "primary",
                }
            ),
            metadata=RoutingPolicyMetadata(
                name=name,
                version=RoutingPolicyVersion.parse("1.0.0"),
            ),
        ),
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )


def test_settings_driven_runtime_uses_default_policy() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(create_active_policy())

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
        ),
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
    )

    assert runtime.refresh_service.policy_name == ("production-routing-policy")

    assert runtime.refresh_service.policy_identifier == (
        "production-routing-policy@1.0.0"
    )


def test_settings_driven_runtime_uses_configured_policy_name() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_active_policy(
            name="enterprise-routing-policy",
        )
    )

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
            llm_routing_policy_name=("enterprise-routing-policy"),
        ),
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
    )

    assert runtime.refresh_service.policy_name == ("enterprise-routing-policy")

    assert runtime.refresh_service.policy_identifier == (
        "enterprise-routing-policy@1.0.0"
    )


def test_settings_driven_runtime_uses_per_request_mode() -> None:
    registry = RoutingPolicyRegistry()

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.PER_REQUEST),
        ),
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
    )

    assert (
        runtime.refresh_service.refresh_config.mode
        is RuntimePolicyRefreshMode.PER_REQUEST
    )

    assert runtime.refresh_service.policy_identifier is None


def test_settings_driven_runtime_uses_ttl_mode() -> None:
    registry = RoutingPolicyRegistry()

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.TTL),
            llm_routing_policy_ttl_seconds=90.0,
        ),
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
    )

    assert runtime.refresh_service.refresh_config.mode is RuntimePolicyRefreshMode.TTL

    assert runtime.refresh_service.refresh_config.ttl_seconds == 90.0


def test_settings_driven_runtime_can_disable_refresh_metrics() -> None:
    registry = RoutingPolicyRegistry()

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.PER_REQUEST),
            llm_routing_refresh_metrics_enabled=False,
        ),
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
    )

    assert runtime.refresh_metrics is None

    assert runtime.refresh_service.refresh_metrics is None


def test_settings_driven_runtime_preserves_routing_metrics() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(create_active_policy())

    routing_metrics = RoutingMetrics()
    exporter = Mock()

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
        ),
        registry=registry,
        models={
            "primary": create_model(
                name="primary",
            ),
        },
        client_factory=Mock(),
        routing_metrics=routing_metrics,
        routing_metrics_exporter=exporter,
    )

    assert runtime.execution_service_factory.metrics is routing_metrics

    assert runtime.execution_service_factory.metrics_exporter is exporter
