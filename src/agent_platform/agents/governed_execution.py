from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.context_budget import (
    ContextFragment,
)


class GovernedAgentExecutionRequest(BaseModel):
    """Input presented to the governed Layer 2 agent runtime."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    user_input: str = Field(
        min_length=1,
    )

    context_fragments: tuple[ContextFragment, ...] = ()

    allowed_citation_ids: frozenset[str] = frozenset()


class GovernedAgentExecutionResult(BaseModel):
    """Final result returned by the governed agent runtime."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    response_text: str = Field(
        min_length=1,
    )

    cited_ids: tuple[str, ...] = ()

    tool_execution_count: int = Field(
        default=0,
        ge=0,
    )


@runtime_checkable
class GovernedAgentExecutionPort(Protocol):
    """Boundary to the governed Layer 2 agent runtime."""

    async def execute(
        self,
        *,
        request: GovernedAgentExecutionRequest,
    ) -> GovernedAgentExecutionResult: ...
