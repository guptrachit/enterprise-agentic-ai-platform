from agent_platform.llm.errors import (
    LLMModelConstraintViolationError,
    LLMModelDisabledError,
    LLMModelNotFoundError,
    LLMModelWorkloadNotSupportedError,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.workload import LLMWorkload


class ModelRouter:
    """Routes LLM workloads to configured model definitions."""

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
        constraints: RoutingConstraints | None = None,
    ) -> ModelDefinition:
        """Resolve and validate the primary model for a workload."""

        return self.route_candidates(
            workload,
            constraints=constraints,
        )[0]

    def route_candidates(
        self,
        workload: LLMWorkload,
        constraints: RoutingConstraints | None = None,
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

            if constraints is not None and not constraints.allows(model):
                continue

            models.append(model)

        if not models:
            primary_model = model_names[0]

            if primary_model not in self.models:
                raise LLMModelNotFoundError(primary_model)

            primary = self.models[primary_model]

            if not primary.enabled:
                raise LLMModelDisabledError(primary_model)

            if not primary.supports_workload(workload):
                raise LLMModelWorkloadNotSupportedError(
                    primary_model,
                    workload.value,
                )

            if constraints is not None:
                raise LLMModelConstraintViolationError(workload.value)

            raise LLMModelWorkloadNotSupportedError(
                primary_model,
                workload.value,
            )

        return tuple(models)
