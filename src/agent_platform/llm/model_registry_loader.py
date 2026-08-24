from agent_platform.config import Settings
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_registry import ModelRegistry


def load_model_registry(
    settings: Settings,
) -> ModelRegistry:
    """Build the runtime model registry from application configuration."""

    models = {
        model.name: ModelDefinition(
            name=model.name,
            provider=model.provider,
            provider_model=model.provider_model,
            workloads=frozenset(model.workloads),
            supports_structured_output=model.supports_structured_output,
            enabled=model.enabled,
            cost_tier=model.cost_tier,
            latency_tier=model.latency_tier,
            capabilities=frozenset(model.capabilities),
        )
        for model in settings.llm_models
    }

    return ModelRegistry(models=models)
