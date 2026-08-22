from enum import StrEnum


class PromptStatus(StrEnum):
    """Lifecycle state of a registered prompt version."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
