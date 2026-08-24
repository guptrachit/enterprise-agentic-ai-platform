from enum import StrEnum


class ModelCapability(StrEnum):
    """Capabilities that may be required from an LLM model."""

    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
