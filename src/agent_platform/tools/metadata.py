from pydantic import BaseModel, ConfigDict, Field


class ToolMetadata(BaseModel):
    """Descriptive metadata associated with a governed platform tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        description="Stable logical identifier for the tool.",
    )

    description: str = Field(
        min_length=1,
        description="Human-readable explanation of what the tool does.",
    )

    version: str = Field(
        min_length=1,
        description="Version of the tool contract or implementation.",
    )
