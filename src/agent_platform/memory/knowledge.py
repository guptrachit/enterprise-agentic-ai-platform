from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeSourceType(StrEnum):
    """Supported logical source categories for enterprise knowledge."""

    DOCUMENT = "document"
    DATABASE = "database"
    API = "api"
    KNOWLEDGE_BASE = "knowledge_base"
    REFERENCE_DATA = "reference_data"


class KnowledgeClassification(StrEnum):
    """Security classification associated with enterprise knowledge."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class EnterpriseDocument(BaseModel):
    """Immutable enterprise knowledge document before chunking."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    document_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)

    title: str = Field(min_length=1)
    content: str = Field(min_length=1)

    source_type: KnowledgeSourceType
    source_uri: str | None = None

    classification: KnowledgeClassification

    version: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None

    tags: frozenset[str] = frozenset()

    attributes: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    """Immutable retrievable unit derived from an enterprise document."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)

    content: str = Field(min_length=1)

    chunk_index: int = Field(ge=0)

    source_type: KnowledgeSourceType
    classification: KnowledgeClassification

    source_uri: str | None = None
    document_title: str | None = None
    document_version: str | None = None

    tags: frozenset[str] = frozenset()

    attributes: dict[str, Any] = Field(default_factory=dict)


class KnowledgeQuery(BaseModel):
    """Provider-neutral request for enterprise knowledge retrieval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: str = Field(min_length=1)

    tenant_id: str = Field(min_length=1)

    top_k: int = Field(
        default=5,
        ge=1,
    )

    allowed_classifications: frozenset[KnowledgeClassification] = frozenset()

    required_tags: frozenset[str] = frozenset()


class RetrievedKnowledgeChunk(BaseModel):
    """Normalized knowledge result returned by retrieval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    chunk: KnowledgeChunk

    rank: int = Field(ge=1)

    score: float | None = None


class KnowledgeRetrievalResult(BaseModel):
    """Normalized enterprise knowledge retrieval result."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: KnowledgeQuery

    items: tuple[RetrievedKnowledgeChunk, ...] = ()
