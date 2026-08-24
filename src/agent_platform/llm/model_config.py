from pydantic import BaseModel, model_validator

from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.workload import LLMWorkload


class ModelConfig(BaseModel):
    """Configuration representation of one logical LLM model."""

    name: str
    provider: str
    provider_model: str
    workloads: tuple[LLMWorkload, ...]
    supports_structured_output: bool = True
    enabled: bool = True
    cost_tier: ModelCostTier = ModelCostTier.MEDIUM
    latency_tier: ModelLatencyTier = ModelLatencyTier.STANDARD
    capabilities: tuple[ModelCapability, ...] | None = None

    @model_validator(mode="after")
    def normalize_capabilities(self) -> "ModelConfig":
        """Normalize the legacy structured-output flag into capabilities."""

        if self.capabilities is None:
            self.capabilities = (
                (ModelCapability.STRUCTURED_OUTPUT,)
                if self.supports_structured_output
                else ()
            )

        self.supports_structured_output = (
            ModelCapability.STRUCTURED_OUTPUT in self.capabilities
        )

        return self
