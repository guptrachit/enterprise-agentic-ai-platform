from enum import StrEnum


class PromptEnvironment(StrEnum):
    """Deployment environment for managed prompts."""

    DEVELOPMENT = "development"
    QA = "qa"
    PRODUCTION = "production"
