from unittest.mock import Mock

from agent_platform.config import Settings
from agent_platform.llm.fully_configured_governed_runtime import (
    create_fully_configured_governed_runtime,
)
from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.workload import LLMWorkload


def create_settings(
    *,
    refresh_mode: RuntimePolicyRefreshMode = (RuntimePolicyRefreshMode.PROCESS_STATIC),
    ttl_seconds: float | None = None,
) -> Settings:
    return Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="primary",
                provider="openai",
                provider_model="primary-provider-model",
                workloads=(LLMWorkload.GENERAL,),
            ),
            ModelConfig(
                name="backup",
                provider="openai",
                provider_model="backup-provider-model",
                workloads=(LLMWorkload.GENERAL,),
            ),
        ),
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.GENERAL: (
                    "primary",
                    "backup",
                ),
            }
        ),
        llm_routing_policy_refresh_mode=refresh_mode,
        llm_routing_policy_ttl_seconds=ttl_seconds,
    )


def test_fully_configured_runtime_builds_registry() -> None:
    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(),
        client_factory=Mock(),
    )

    active = runtime.registry.get_active("production-routing-policy")

    assert active is not None

    assert active.status is RoutingPolicyLifecycleStatus.ACTIVE

    assert active.identifier == ("production-routing-policy@1.0.0")


def test_fully_configured_runtime_resolves_default_policy() -> None:
    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(),
        client_factory=Mock(),
    )

    assert runtime.refresh_service.policy_identifier == (
        "production-routing-policy@1.0.0"
    )


def test_fully_configured_runtime_builds_models_from_settings() -> None:
    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(),
        client_factory=Mock(),
    )

    router = runtime.router_factory.create("production-routing-policy")

    decision = router.route_decision(LLMWorkload.GENERAL)

    assert decision.selected_model.name == "primary"


def test_fully_configured_runtime_preserves_fallback_order() -> None:
    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(),
        client_factory=Mock(),
    )

    router = runtime.router_factory.create("production-routing-policy")

    decision = router.route_decision(LLMWorkload.GENERAL)

    assert tuple(model.name for model in decision.ranked_candidates) == (
        "primary",
        "backup",
    )


def test_fully_configured_runtime_supports_per_request_refresh() -> None:
    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(
            refresh_mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
        client_factory=Mock(),
    )

    assert runtime.refresh_service.policy_identifier is None

    assert (
        runtime.refresh_service.refresh_config.mode
        is RuntimePolicyRefreshMode.PER_REQUEST
    )


def test_fully_configured_runtime_supports_ttl_refresh() -> None:
    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(
            refresh_mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=90.0,
        ),
        client_factory=Mock(),
    )

    assert runtime.refresh_service.refresh_config.mode is RuntimePolicyRefreshMode.TTL

    assert runtime.refresh_service.refresh_config.ttl_seconds == 90.0
