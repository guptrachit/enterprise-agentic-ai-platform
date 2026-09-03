from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSelection(BaseModel):
    """A tool invocation proposed by an LLM."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tool_name: str = Field(
        min_length=1,
        description="Logical name of the selected governed tool.",
    )

    tool_version: str = Field(
        min_length=1,
        description="Version of the selected governed tool.",
    )

    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments proposed by the LLM for tool execution.",
    )


class ToolSelectionRequest(BaseModel):
    """Input supplied to an LLM tool-selection component."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    user_message: str = Field(
        min_length=1,
        description="User request from which the LLM selects a tool.",
    )


class ToolSelectionError(Exception):
    """Base error raised during LLM tool selection."""


class ToolSelectionNotFoundError(ToolSelectionError):
    """Raised when the LLM does not select a valid tool."""
