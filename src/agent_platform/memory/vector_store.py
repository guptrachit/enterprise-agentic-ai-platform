from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.embeddings import EmbeddingVector
from agent_platform.memory.knowledge import (
    KnowledgeChunk,
    KnowledgeClassification,
)


class VectorRecord(BaseModel):
    """Stored vector plus its governed enterprise knowledge chunk."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    vector_id: str = Field(min_length=1)
    chunk: KnowledgeChunk
    vector: EmbeddingVector


class VectorSearchQuery(BaseModel):
    """Provider-neutral vector similarity search request."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    vector: EmbeddingVector
    tenant_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
    required_tags: frozenset[str] = frozenset()
    allowed_classifications: frozenset[KnowledgeClassification] = frozenset()


class VectorSearchResultItem(BaseModel):
    """Single normalized vector search result."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    record: VectorRecord
    score: float
    rank: int = Field(ge=1)


class VectorSearchResult(BaseModel):
    """Normalized vector-search result."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: VectorSearchQuery
    items: tuple[VectorSearchResultItem, ...] = ()


@runtime_checkable
class VectorStore(Protocol):
    """Provider-neutral vector persistence and search contract."""

    async def upsert(
        self,
        *,
        record: VectorRecord,
    ) -> VectorRecord:
        """Insert or replace a vector record."""
        ...

    async def get(
        self,
        *,
        vector_id: str,
    ) -> VectorRecord | None:
        """Return one stored vector by exact identifier."""
        ...

    async def search(
        self,
        *,
        query: VectorSearchQuery,
    ) -> VectorSearchResult:
        """Return semantically similar vectors."""
        ...

    async def delete(
        self,
        *,
        vector_id: str,
    ) -> bool:
        """Delete a vector record."""
        ...


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""
