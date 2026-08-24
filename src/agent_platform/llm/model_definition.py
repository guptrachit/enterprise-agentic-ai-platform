from dataclasses import dataclass

from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.workload import LLMWorkload


@dataclass(frozen=True)
class ModelDefinition:
    """Platform definition of an available LLM model."""

    name: str
    provider: str
    provider_model: str
    workloads: frozenset[LLMWorkload]
    supports_structured_output: bool = True
    enabled: bool = True
    cost_tier: ModelCostTier = ModelCostTier.MEDIUM
    latency_tier: ModelLatencyTier = ModelLatencyTier.STANDARD
    capabilities: frozenset[ModelCapability] | None = None

    def __post_init__(self) -> None:
        """Normalize legacy capability fields into the capability contract."""

        if self.capabilities is None:
            capabilities = (
                frozenset(
                    {
                        ModelCapability.STRUCTURED_OUTPUT,
                    }
                )
                if self.supports_structured_output
                else frozenset()
            )
        else:
            capabilities = self.capabilities

        object.__setattr__(
            self,
            "capabilities",
            capabilities,
        )

        object.__setattr__(
            self,
            "supports_structured_output",
            ModelCapability.STRUCTURED_OUTPUT in capabilities,
        )

    def supports_workload(
        self,
        workload: LLMWorkload,
    ) -> bool:
        """Return whether this model supports the requested workload."""

        return workload in self.workloads

    def supports_capabilities(
        self,
        required: frozenset[ModelCapability],
    ) -> bool:
        """Return whether the model supports all required capabilities."""

        capabilities = self.capabilities or frozenset()

        return required.issubset(capabilities)
