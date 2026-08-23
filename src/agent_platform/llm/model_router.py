from agent_platform.llm.errors import (
    LLMModelDisabledError,
    LLMModelNotFoundError,
    LLMModelWorkloadNotSupportedError,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.workload import LLMWorkload


class ModelRouter:
    """Resolve workloads to validated model definitions."""

    def __init__(
        self,
        models: dict[str, ModelDefinition],
        policy: ModelPolicy,
    ) -> None:
        self.models = models
        self.policy = policy

    def route(
        self,
        workload: LLMWorkload,
    ) -> ModelDefinition:
        """Resolve and validate the model for a workload."""

        return self.route_candidates(workload)[0]

    def route_candidates(
        self,
        workload: LLMWorkload,
    ) -> tuple[ModelDefinition, ...]:
        """Resolve all valid model candidates for a workload."""

        model_names = self.policy.models_for(workload)

        models: list[ModelDefinition] = []

        for model_name in model_names:
            try:
                model = self.models[model_name]
            except KeyError as error:
                raise LLMModelNotFoundError(model_name) from error

            if not model.enabled:
                continue

            if not model.supports_workload(workload):
                continue

            models.append(model)

        if not models:
            primary_model = model_names[0]

            if primary_model not in self.models:
                raise LLMModelNotFoundError(primary_model)

            if not self.models[primary_model].enabled:
                raise LLMModelDisabledError(primary_model)

            raise LLMModelWorkloadNotSupportedError(
                primary_model,
                workload.value,
            )

        return tuple(models)
