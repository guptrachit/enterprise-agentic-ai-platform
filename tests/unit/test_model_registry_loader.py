from agent_platform.config import Settings
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_registry_loader import load_model_registry
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.workload import LLMWorkload


def test_load_model_registry_from_settings() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="fast_general",
                provider="openai",
                provider_model="gpt-5-mini",
                workloads=(
                    LLMWorkload.GENERAL,
                    LLMWorkload.CLASSIFICATION,
                ),
                cost_tier=ModelCostTier.LOW,
                latency_tier=ModelLatencyTier.FAST,
                capabilities=(
                    ModelCapability.STRUCTURED_OUTPUT,
                    ModelCapability.TOOL_CALLING,
                ),
            ),
            ModelConfig(
                name="reasoning_model",
                provider="openai",
                provider_model="reasoning-model",
                workloads=(LLMWorkload.REASONING,),
                cost_tier=ModelCostTier.HIGH,
                latency_tier=ModelLatencyTier.SLOW,
                capabilities=(ModelCapability.STRUCTURED_OUTPUT,),
            ),
        ),
    )

    registry = load_model_registry(settings)

    fast = registry.get("fast_general")
    reasoning = registry.get("reasoning_model")

    assert fast.provider == "openai"
    assert fast.provider_model == "gpt-5-mini"

    assert fast.workloads == frozenset(
        {
            LLMWorkload.GENERAL,
            LLMWorkload.CLASSIFICATION,
        }
    )

    assert fast.cost_tier is ModelCostTier.LOW
    assert fast.latency_tier is ModelLatencyTier.FAST

    assert fast.capabilities == frozenset(
        {
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.TOOL_CALLING,
        }
    )

    assert reasoning.provider_model == "reasoning-model"
    assert reasoning.cost_tier is ModelCostTier.HIGH
    assert reasoning.latency_tier is ModelLatencyTier.SLOW

    assert reasoning.capabilities == frozenset(
        {
            ModelCapability.STRUCTURED_OUTPUT,
        }
    )


def test_load_model_registry_with_no_configured_models() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    registry = load_model_registry(settings)

    assert registry.all() == ()
