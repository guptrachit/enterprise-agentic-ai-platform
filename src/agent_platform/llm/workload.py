from enum import StrEnum


class LLMWorkload(StrEnum):
    """Logical workload categories used for model policy and routing."""

    GENERAL = "general"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
