from dataclasses import dataclass

from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)


@dataclass(frozen=True)
class ModelPreference:
    """Preferences used to rank otherwise eligible model candidates."""

    prefer_lower_cost: bool = False
    prefer_lower_latency: bool = False
    preferred_providers: tuple[str, ...] = ()
    preferred_cost_tier: ModelCostTier | None = None
    preferred_latency_tier: ModelLatencyTier | None = None
