from dataclasses import dataclass

from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.routing_reason import RoutingReason


@dataclass(frozen=True)
class RoutingDecision:
    """Explains the outcome of an LLM model routing decision."""

    selected_model: ModelDefinition
    ranked_candidates: tuple[ModelDefinition, ...]
    rejected_models: tuple[str, ...] = ()
    reasons: tuple[RoutingReason, ...] = ()
