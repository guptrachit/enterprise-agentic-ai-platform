from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.workload import LLMWorkload


def test_routing_constraints_allow_matching_model() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        cost_tier=ModelCostTier.LOW,
        latency_tier=ModelLatencyTier.FAST,
    )

    constraints = RoutingConstraints(
        allowed_providers=frozenset(
            {
                "openai",
            }
        ),
        max_cost_tier=ModelCostTier.LOW,
        max_latency_tier=ModelLatencyTier.FAST,
    )

    assert constraints.allows(model)


def test_routing_constraints_reject_disallowed_provider() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )

    constraints = RoutingConstraints(
        allowed_providers=frozenset(
            {
                "anthropic",
            }
        )
    )

    assert not constraints.allows(model)


def test_routing_constraints_reject_expensive_model() -> None:
    model = ModelDefinition(
        name="reasoning_model",
        provider="openai",
        provider_model="reasoning-model",
        workloads=frozenset(
            {
                LLMWorkload.REASONING,
            }
        ),
        cost_tier=ModelCostTier.HIGH,
    )

    constraints = RoutingConstraints(max_cost_tier=ModelCostTier.MEDIUM)

    assert not constraints.allows(model)


def test_routing_constraints_reject_slow_model() -> None:
    model = ModelDefinition(
        name="slow_model",
        provider="openai",
        provider_model="slow-model",
        workloads=frozenset(
            {
                LLMWorkload.REASONING,
            }
        ),
        latency_tier=ModelLatencyTier.SLOW,
    )

    constraints = RoutingConstraints(max_latency_tier=ModelLatencyTier.STANDARD)

    assert not constraints.allows(model)
