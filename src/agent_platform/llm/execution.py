from dataclasses import dataclass

from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.workload import LLMWorkload


@dataclass(frozen=True)
class LLMExecutionRequest:
    """Provider-neutral request for LLM execution."""

    prompt: str
    workload: LLMWorkload = LLMWorkload.GENERAL
    correlation_id: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    constraints: RoutingConstraints | None = None
    required_capabilities: frozenset[ModelCapability] = frozenset()
