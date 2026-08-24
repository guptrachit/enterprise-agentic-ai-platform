from agent_platform.config import Settings
from agent_platform.llm.model_policy import ModelPolicy


def load_model_policy(
    settings: Settings,
) -> ModelPolicy:
    """Build the runtime model policy from application configuration."""

    return ModelPolicy(assignments=dict(settings.llm_model_policy.assignments))
