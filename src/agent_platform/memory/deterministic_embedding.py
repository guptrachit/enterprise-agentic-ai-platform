from agent_platform.memory.embeddings import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingVector,
)


class DeterministicEmbeddingProvider:
    """Simple deterministic embedding provider for development and tests."""

    def __init__(
        self,
        *,
        dimensions: int = 4,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")

        self._dimensions = dimensions

    async def embed(
        self,
        *,
        request: EmbeddingRequest,
    ) -> EmbeddingResult:
        normalized = request.text.casefold().encode("utf-8")

        buckets = [0.0 for _ in range(self._dimensions)]

        for index, byte_value in enumerate(normalized):
            bucket_index = index % self._dimensions

            buckets[bucket_index] += byte_value / 255.0

        magnitude = sum(value * value for value in buckets) ** 0.5

        if magnitude > 0:
            buckets = [value / magnitude for value in buckets]

        return EmbeddingResult(
            vector=EmbeddingVector(
                values=tuple(buckets),
            ),
            provider="deterministic",
            model=(f"deterministic-{self._dimensions}d"),
        )
