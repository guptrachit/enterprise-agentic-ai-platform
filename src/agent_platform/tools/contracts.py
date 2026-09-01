from pydantic import BaseModel, ConfigDict, Field


class ToolExecutionContext(BaseModel):
    """Platform context carried with every tool execution."""

    model_config = ConfigDict(frozen=True)

    correlation_id: str = Field(
        min_length=1,
        description="Correlation identifier used to trace the tool execution.",
    )


class ToolInput(BaseModel):
    """Base contract for tool inputs."""

    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    """Base contract for tool outputs."""

    model_config = ConfigDict(extra="forbid")
