from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)


def test_model_preference_defaults() -> None:
    preference = ModelPreference()

    assert preference.prefer_lower_cost is False
    assert preference.prefer_lower_latency is False
    assert preference.preferred_providers == ()
    assert preference.preferred_cost_tier is None
    assert preference.preferred_latency_tier is None


def test_model_preference_preserves_values() -> None:
    preference = ModelPreference(
        prefer_lower_cost=True,
        prefer_lower_latency=True,
        preferred_providers=(
            "openai",
            "anthropic",
        ),
        preferred_cost_tier=ModelCostTier.LOW,
        preferred_latency_tier=ModelLatencyTier.FAST,
    )

    assert preference.prefer_lower_cost is True
    assert preference.prefer_lower_latency is True

    assert preference.preferred_providers == (
        "openai",
        "anthropic",
    )

    assert preference.preferred_cost_tier is ModelCostTier.LOW
    assert preference.preferred_latency_tier is ModelLatencyTier.FAST
