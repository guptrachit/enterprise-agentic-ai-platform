from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.tools.selection import ToolSelection


class AgentObservation(BaseModel):
    """Result returned to the agent after governed tool execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tool_name: str = Field(
        min_length=1,
        description="Logical name of the executed tool.",
    )

    tool_version: str = Field(
        min_length=1,
        description="Version of the executed tool.",
    )

    output: dict[str, Any] = Field(
        description="Validated tool output returned to the model.",
    )


class AgentFinalResponse(BaseModel):
    """Final response produced by the agent for the user."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    content: str = Field(
        min_length=1,
        description="Final user-facing response.",
    )


class AgentDecision(BaseModel):
    """Single decision produced by the agent model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tool_selection: ToolSelection | None = None
    final_response: AgentFinalResponse | None = None

    def model_post_init(self, __context: Any, /) -> None:
        has_tool = self.tool_selection is not None
        has_final = self.final_response is not None

        if has_tool == has_final:
            raise ValueError(
                "agent decision must contain exactly one of "
                "tool_selection or final_response"
            )


class AgentApprovalRequired(BaseModel):
    """Returned when agent execution pauses for human approval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    approval_id: str = Field(
        min_length=1,
    )

    tool_selection: ToolSelection
