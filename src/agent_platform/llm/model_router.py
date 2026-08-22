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

        model_name = self.policy.model_for(workload)

        try:
            model = self.models[model_name]
        except KeyError as error:
            raise LLMModelNotFoundError(model_name) from error

        if not model.enabled:
            raise LLMModelDisabledError(model.name)

        if not model.supports_workload(workload):
            raise LLMModelWorkloadNotSupportedError(
                model.name,
                workload.value,
            )

        return model
