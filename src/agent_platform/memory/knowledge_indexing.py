from agent_platform.memory.embedding_service import EmbeddingService
from agent_platform.memory.knowledge import KnowledgeChunk
from agent_platform.memory.vector_store import (
    VectorRecord,
    VectorStore,
)


class KnowledgeIndexingService:
    """Creates embeddings for knowledge chunks and stores them as vectors."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    async def index_chunk(
        self,
        *,
        chunk: KnowledgeChunk,
    ) -> VectorRecord:
        """Embed and persist one enterprise knowledge chunk."""

        embedding = await self._embedding_service.embed_text(
            text=chunk.content,
        )

        record = VectorRecord(
            vector_id=chunk.chunk_id,
            chunk=chunk,
            vector=embedding.vector,
        )

        return await self._vector_store.upsert(
            record=record,
        )

    async def index_many(
        self,
        *,
        chunks: tuple[KnowledgeChunk, ...],
    ) -> tuple[VectorRecord, ...]:
        """Index multiple chunks while preserving input order."""

        records: list[VectorRecord] = []

        for chunk in chunks:
            records.append(
                await self.index_chunk(
                    chunk=chunk,
                )
            )

        return tuple(records)
