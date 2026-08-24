from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_preference import ModelPreference


class ModelRanker:
    """Ranks eligible LLM model candidates using soft preferences."""

    def rank(
        self,
        models: tuple[ModelDefinition, ...],
        preference: ModelPreference | None = None,
    ) -> tuple[ModelDefinition, ...]:
        """Return models ordered by preference score."""

        if preference is None:
            return models

        indexed_models = tuple(enumerate(models))

        ranked = sorted(
            indexed_models,
            key=lambda item: (
                self._score(
                    item[1],
                    preference,
                ),
                -item[0],
            ),
            reverse=True,
        )

        return tuple(model for _, model in ranked)

    def _score(
        self,
        model: ModelDefinition,
        preference: ModelPreference,
    ) -> int:
        """Calculate a relative ranking score for one model."""

        score = 0

        if preference.prefer_lower_cost:
            score += -int(model.cost_tier)

        if preference.prefer_lower_latency:
            score += -int(model.latency_tier)

        if preference.preferred_cost_tier is not None:
            if model.cost_tier == preference.preferred_cost_tier:
                score += 10

        if preference.preferred_latency_tier is not None:
            if model.latency_tier == preference.preferred_latency_tier:
                score += 10

        if preference.preferred_providers:
            try:
                provider_index = preference.preferred_providers.index(model.provider)
            except ValueError:
                pass
            else:
                score += (len(preference.preferred_providers) - provider_index) * 10

        return score
