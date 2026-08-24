from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.model_ranker import ModelRanker
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
    provider: str = "openai",
    cost_tier: ModelCostTier = ModelCostTier.MEDIUM,
    latency_tier: ModelLatencyTier = ModelLatencyTier.STANDARD,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider=provider,
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        cost_tier=cost_tier,
        latency_tier=latency_tier,
    )


def test_ranker_preserves_order_without_preference() -> None:
    first = create_model(
        name="first",
    )
    second = create_model(
        name="second",
    )

    ranker = ModelRanker()

    assert ranker.rank(
        (
            first,
            second,
        )
    ) == (
        first,
        second,
    )


def test_ranker_prefers_lower_cost() -> None:
    expensive = create_model(
        name="expensive",
        cost_tier=ModelCostTier.HIGH,
    )

    cheap = create_model(
        name="cheap",
        cost_tier=ModelCostTier.LOW,
    )

    ranker = ModelRanker()

    result = ranker.rank(
        (
            expensive,
            cheap,
        ),
        ModelPreference(
            prefer_lower_cost=True,
        ),
    )

    assert result == (
        cheap,
        expensive,
    )


def test_ranker_prefers_lower_latency() -> None:
    slow = create_model(
        name="slow",
        latency_tier=ModelLatencyTier.SLOW,
    )

    fast = create_model(
        name="fast",
        latency_tier=ModelLatencyTier.FAST,
    )

    ranker = ModelRanker()

    result = ranker.rank(
        (
            slow,
            fast,
        ),
        ModelPreference(
            prefer_lower_latency=True,
        ),
    )

    assert result == (
        fast,
        slow,
    )


def test_ranker_prefers_exact_cost_tier() -> None:
    medium = create_model(
        name="medium",
        cost_tier=ModelCostTier.MEDIUM,
    )

    low = create_model(
        name="low",
        cost_tier=ModelCostTier.LOW,
    )

    ranker = ModelRanker()

    result = ranker.rank(
        (
            medium,
            low,
        ),
        ModelPreference(
            preferred_cost_tier=ModelCostTier.LOW,
        ),
    )

    assert result[0] is low


def test_ranker_prefers_exact_latency_tier() -> None:
    standard = create_model(
        name="standard",
        latency_tier=ModelLatencyTier.STANDARD,
    )

    fast = create_model(
        name="fast",
        latency_tier=ModelLatencyTier.FAST,
    )

    ranker = ModelRanker()

    result = ranker.rank(
        (
            standard,
            fast,
        ),
        ModelPreference(
            preferred_latency_tier=ModelLatencyTier.FAST,
        ),
    )

    assert result[0] is fast


def test_ranker_respects_provider_preference_order() -> None:
    second_provider = create_model(
        name="second_provider",
        provider="anthropic",
    )

    preferred_provider = create_model(
        name="preferred_provider",
        provider="openai",
    )

    ranker = ModelRanker()

    result = ranker.rank(
        (
            second_provider,
            preferred_provider,
        ),
        ModelPreference(
            preferred_providers=(
                "openai",
                "anthropic",
            ),
        ),
    )

    assert result == (
        preferred_provider,
        second_provider,
    )


def test_ranker_preserves_policy_order_on_equal_score() -> None:
    first = create_model(
        name="first",
    )

    second = create_model(
        name="second",
    )

    ranker = ModelRanker()

    result = ranker.rank(
        (
            first,
            second,
        ),
        ModelPreference(
            prefer_lower_cost=True,
            prefer_lower_latency=True,
        ),
    )

    assert result == (
        first,
        second,
    )
