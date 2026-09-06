from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ContextSourceType(StrEnum):
    CONVERSATION = "conversation"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    ENTERPRISE_KNOWLEDGE = "enterprise_knowledge"


class ContextBudget(BaseModel):
    """Maximum context budget available for model input."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    max_tokens: int = Field(
        ge=1,
    )


class ContextFragment(BaseModel):
    """One normalized piece of context before final prompt assembly."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    content: str = Field(
        min_length=1,
    )

    source_type: ContextSourceType

    source_id: str | None = None

    citation_id: str | None = None

    estimated_tokens: int = Field(
        ge=1,
    )

    priority: int = Field(
        default=100,
        ge=0,
    )


class ContextAssemblyResult(BaseModel):
    """Final bounded context selected for model consumption."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    fragments: tuple[ContextFragment, ...] = ()

    used_tokens: int = Field(
        ge=0,
    )

    max_tokens: int = Field(
        ge=1,
    )

    truncated: bool = False


@runtime_checkable
class TokenEstimator(Protocol):
    """Provider-neutral token estimation contract."""

    def estimate(
        self,
        *,
        text: str,
    ) -> int:
        """Estimate token usage for supplied text."""
        ...


class ContextBudgetExceededError(Exception):
    """Raised when required context cannot fit within the token budget."""
