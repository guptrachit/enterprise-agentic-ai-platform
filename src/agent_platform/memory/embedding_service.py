from agent_platform.memory.embeddings import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResult,
)


class EmbeddingService:
    """Platform-owned entry point for provider-neutral embeddings."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
    ) -> None:
        self._provider = provider

    async def embed_text(
        self,
        *,
        text: str,
    ) -> EmbeddingResult:
        """Generate an embedding for a single text value."""

        return await self._provider.embed(
            request=EmbeddingRequest(
                text=text,
            )
        )

    async def embed_many(
        self,
        *,
        texts: tuple[str, ...],
    ) -> tuple[EmbeddingResult, ...]:
        """Generate embeddings while preserving input order."""

        results: list[EmbeddingResult] = []

        for text in texts:
            results.append(
                await self.embed_text(
                    text=text,
                )
            )

        return tuple(results)
