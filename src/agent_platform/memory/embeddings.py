from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRequest(BaseModel):
    """Provider-neutral request for vector generation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    text: str = Field(min_length=1)


class EmbeddingVector(BaseModel):
    """Normalized vector returned by an embedding provider."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    values: tuple[float, ...] = Field(min_length=1)

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vector."""
        return len(self.values)


class EmbeddingResult(BaseModel):
    """Normalized embedding response with provider metadata."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    vector: EmbeddingVector

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-neutral contract for text embedding generation."""

    async def embed(
        self,
        *,
        request: EmbeddingRequest,
    ) -> EmbeddingResult:
        """Generate a vector representation of the supplied text."""
        ...


class EmbeddingProviderError(Exception):
    """Raised when an embedding provider cannot generate a vector."""
