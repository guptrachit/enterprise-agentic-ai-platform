from agent_platform.llm.errors import (
    LLMModelConstraintViolationError,
    LLMModelDisabledError,
    LLMModelWorkloadNotSupportedError,
)
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_registry import ModelRegistry
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.workload import LLMWorkload


class ModelRouter:
    """Routes logical LLM workloads to valid model definitions."""

    def __init__(
        self,
        models: dict[str, ModelDefinition] | None = None,
        policy: ModelPolicy | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        if registry is not None:
            self.registry = registry
        else:
            self.registry = ModelRegistry(models=models)

        if policy is None:
            raise ValueError("Model policy is required.")

        self.policy = policy

    def route(
        self,
        workload: LLMWorkload,
        constraints: RoutingConstraints | None = None,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
    ) -> ModelDefinition:
        """Resolve the primary valid model for a workload."""

        return self.route_candidates(
            workload,
            constraints=constraints,
            required_capabilities=required_capabilities,
        )[0]

    def route_candidates(
        self,
        workload: LLMWorkload,
        constraints: RoutingConstraints | None = None,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
    ) -> tuple[ModelDefinition, ...]:
        """Resolve all valid model candidates for a workload."""

        model_names = self.policy.models_for(workload)

        models: list[ModelDefinition] = []

        for model_name in model_names:
            model = self.registry.get(model_name)

            if not model.enabled:
                continue

            if not model.supports_workload(workload):
                continue

            if constraints is not None and not constraints.allows(model):
                continue

            if not model.supports_capabilities(required_capabilities):
                continue

            models.append(model)

        if not models:
            primary_model = model_names[0]

            primary = self.registry.get(primary_model)

            if not primary.enabled:
                raise LLMModelDisabledError(primary_model)

            if not primary.supports_workload(workload):
                raise LLMModelWorkloadNotSupportedError(
                    primary_model,
                    workload.value,
                )

            if constraints is not None and not constraints.allows(primary):
                raise LLMModelConstraintViolationError(workload.value)

            if not primary.supports_capabilities(required_capabilities):
                raise LLMModelConstraintViolationError(workload.value)

            raise LLMModelWorkloadNotSupportedError(
                primary_model,
                workload.value,
            )

        return tuple(models)
