from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.workload import LLMWorkload


def test_model_config_parses_model_configuration() -> None:
    config = ModelConfig(
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
    )

    assert config.name == "fast_general"
    assert config.provider == "openai"
    assert config.provider_model == "gpt-5-mini"

    assert LLMWorkload.CLASSIFICATION in config.workloads

    assert config.cost_tier is ModelCostTier.LOW
    assert config.latency_tier is ModelLatencyTier.FAST

    assert config.capabilities == (
        ModelCapability.STRUCTURED_OUTPUT,
        ModelCapability.TOOL_CALLING,
    )

    assert config.supports_structured_output is True
    assert config.enabled is True


def test_model_config_preserves_legacy_structured_output_default() -> None:
    config = ModelConfig(
        name="legacy_model",
        provider="openai",
        provider_model="legacy-model",
        workloads=(LLMWorkload.GENERAL,),
    )

    assert config.capabilities == (ModelCapability.STRUCTURED_OUTPUT,)

    assert config.supports_structured_output is True


def test_explicit_capabilities_are_authoritative() -> None:
    config = ModelConfig(
        name="tool_only",
        provider="openai",
        provider_model="tool-model",
        workloads=(LLMWorkload.GENERAL,),
        supports_structured_output=True,
        capabilities=(ModelCapability.TOOL_CALLING,),
    )

    assert config.capabilities == (ModelCapability.TOOL_CALLING,)

    assert config.supports_structured_output is False


def test_explicit_structured_output_capability_enables_legacy_flag() -> None:
    config = ModelConfig(
        name="structured_model",
        provider="openai",
        provider_model="structured-model",
        workloads=(LLMWorkload.GENERAL,),
        supports_structured_output=False,
        capabilities=(ModelCapability.STRUCTURED_OUTPUT,),
    )

    assert config.supports_structured_output is True
