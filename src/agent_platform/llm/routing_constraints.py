from dataclasses import dataclass

from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)


@dataclass(frozen=True)
class RoutingConstraints:
    """Optional constraints applied to routed model candidates."""

    allowed_providers: frozenset[str] | None = None
    max_cost_tier: ModelCostTier | None = None
    max_latency_tier: ModelLatencyTier | None = None

    def allows(
        self,
        model: ModelDefinition,
    ) -> bool:
        """Return whether a model satisfies all routing constraints."""

        if (
            self.allowed_providers is not None
            and model.provider not in self.allowed_providers
        ):
            return False

        if self.max_cost_tier is not None and model.cost_tier > self.max_cost_tier:
            return False

        return not (
            self.max_latency_tier is not None
            and model.latency_tier > self.max_latency_tier
        )
