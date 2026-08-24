from agent_platform.config import Settings
from agent_platform.llm.errors import LLMConfigurationError


def validate_model_configuration(
    settings: Settings,
) -> None:
    """Validate consistency between configured models and routing policy."""

    configured_models = settings.llm_models

    model_names = [model.name for model in configured_models]

    duplicate_names = sorted(
        {name for name in model_names if model_names.count(name) > 1}
    )

    if duplicate_names:
        duplicates = ", ".join(duplicate_names)

        raise LLMConfigurationError(
            f"Duplicate logical model names configured: {duplicates}"
        )

    models_by_name = {model.name: model for model in configured_models}

    for model in configured_models:
        capabilities = model.capabilities or ()

        if len(capabilities) != len(set(capabilities)):
            raise LLMConfigurationError(
                f"Duplicate capabilities configured for model '{model.name}'"
            )

    for workload, assignment in settings.llm_model_policy.assignments.items():
        model_names_for_workload = (
            (assignment,) if isinstance(assignment, str) else assignment
        )

        for model_name in model_names_for_workload:
            if model_name not in models_by_name:
                raise LLMConfigurationError(
                    "Model policy references undefined model "
                    f"'{model_name}' for workload '{workload.value}'"
                )

            model = models_by_name[model_name]

            if workload not in model.workloads:
                raise LLMConfigurationError(
                    f"Model '{model_name}' does not support workload '{workload.value}'"
                )
