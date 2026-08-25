from agent_platform.config import Settings
from agent_platform.llm.configured_routing_policy_registry import (
    create_configured_routing_policy_registry,
)
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.workload import LLMWorkload


def test_configured_registry_creates_active_policy() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.GENERAL: (
                    "primary",
                    "backup",
                ),
            }
        ),
    )

    registry = create_configured_routing_policy_registry(settings)

    active = registry.get_active("production-routing-policy")

    assert active is not None

    assert active.status is RoutingPolicyLifecycleStatus.ACTIVE

    assert active.identifier == ("production-routing-policy@1.0.0")


def test_configured_registry_preserves_policy_assignments() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.GENERAL: (
                    "primary",
                    "backup",
                ),
            }
        ),
    )

    registry = create_configured_routing_policy_registry(settings)

    active = registry.get_active("production-routing-policy")

    assert active is not None

    assert active.versioned_policy.policy.assignments[LLMWorkload.GENERAL] == (
        "primary",
        "backup",
    )


def test_configured_registry_uses_configured_policy_name() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_routing_policy_name="enterprise-routing-policy",
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    registry = create_configured_routing_policy_registry(settings)

    active = registry.get_active("enterprise-routing-policy")

    assert active is not None

    assert active.identifier == ("enterprise-routing-policy@1.0.0")


def test_configured_registry_contains_single_policy() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    registry = create_configured_routing_policy_registry(settings)

    assert len(registry) == 1


def test_configured_registry_policy_has_bootstrap_description() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    registry = create_configured_routing_policy_registry(settings)

    active = registry.get_active("production-routing-policy")

    assert active is not None

    assert active.versioned_policy.metadata.description == (
        "Routing policy bootstrapped from application settings."
    )
