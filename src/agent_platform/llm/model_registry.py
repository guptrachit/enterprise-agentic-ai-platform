from agent_platform.llm.errors import LLMModelNotFoundError
from agent_platform.llm.model_definition import ModelDefinition


class ModelRegistry:
    """Central registry of logical LLM model definitions."""

    def __init__(
        self,
        models: dict[str, ModelDefinition] | None = None,
    ) -> None:
        self._models: dict[str, ModelDefinition] = dict(models or {})

    def register(
        self,
        model: ModelDefinition,
    ) -> None:
        """Register or replace a logical model definition."""

        self._models[model.name] = model

    def get(
        self,
        name: str,
    ) -> ModelDefinition:
        """Return a registered model definition."""

        try:
            return self._models[name]
        except KeyError as error:
            raise LLMModelNotFoundError(name) from error

    def all(
        self,
    ) -> tuple[ModelDefinition, ...]:
        """Return all registered model definitions."""

        return tuple(self._models.values())

    def as_dict(
        self,
    ) -> dict[str, ModelDefinition]:
        """Return a copy of registered models keyed by logical name."""

        return dict(self._models)
