from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryType(StrEnum):
    """Logical category of memory stored by the platform."""

    CONVERSATION = "conversation"
    WORKING = "working"
    LONG_TERM = "long_term"
    ENTERPRISE_KNOWLEDGE = "enterprise_knowledge"


class MemoryScope(StrEnum):
    """Visibility boundary applied to stored memory."""

    EXECUTION = "execution"
    CONVERSATION = "conversation"
    USER = "user"
    TENANT = "tenant"
    GLOBAL = "global"


class ContextItem(BaseModel):
    """Base unit of information supplied to an agent or model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    content: str = Field(min_length=1)
    source: str | None = None


class MemoryMetadata(BaseModel):
    """Metadata used to govern and retrieve persisted memory."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    memory_type: MemoryType
    scope: MemoryScope
    subject_id: str | None = None
    tenant_id: str | None = None
    conversation_id: str | None = None
    execution_id: str | None = None
    tags: frozenset[str] = frozenset()


class MemoryRecord(BaseModel):
    """Immutable memory object managed by the platform."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    memory_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: MemoryMetadata
    created_at: datetime
    attributes: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """Provider-neutral request for memory or knowledge retrieval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
    subject_id: str | None = None
    tenant_id: str | None = None
    conversation_id: str | None = None
    required_tags: frozenset[str] = frozenset()


class RetrievedItem(BaseModel):
    """Single item returned from a governed retrieval operation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    record: MemoryRecord
    score: float | None = None
    rank: int = Field(ge=1)


class RetrievalResult(BaseModel):
    """Normalized result returned by a retrieval provider."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: RetrievalQuery
    items: tuple[RetrievedItem, ...] = ()
