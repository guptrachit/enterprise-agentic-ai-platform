from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.knowledge import (
    KnowledgeQuery,
    RetrievedKnowledgeChunk,
)


class RerankRequest(BaseModel):
    """Provider-neutral request for reranking retrieved knowledge."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: KnowledgeQuery

    candidates: tuple[RetrievedKnowledgeChunk, ...]

    top_k: int = Field(
        default=5,
        ge=1,
    )


class RerankedKnowledgeChunk(BaseModel):
    """One candidate after reranking."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    item: RetrievedKnowledgeChunk

    rerank_score: float

    rank: int = Field(
        ge=1,
    )


class RerankResult(BaseModel):
    """Normalized reranking result."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: KnowledgeQuery

    items: tuple[RerankedKnowledgeChunk, ...] = ()


@runtime_checkable
class Reranker(Protocol):
    """Provider-neutral reranking contract."""

    async def rerank(
        self,
        *,
        request: RerankRequest,
    ) -> RerankResult:
        """Reorder candidate chunks by final relevance."""
        ...


class RerankerError(Exception):
    """Raised when reranking cannot be completed."""
