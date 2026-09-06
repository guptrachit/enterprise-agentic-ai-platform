from agent_platform.memory.embedding_service import EmbeddingService
from agent_platform.memory.knowledge import (
    KnowledgeQuery,
    KnowledgeRetrievalResult,
    RetrievedKnowledgeChunk,
)
from agent_platform.memory.vector_store import (
    VectorSearchQuery,
    VectorStore,
)


class SemanticRetrievalService:
    """Provider-neutral semantic retrieval orchestration service."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    async def retrieve(
        self,
        *,
        query: KnowledgeQuery,
    ) -> KnowledgeRetrievalResult:
        """Embed the query, perform vector search, and normalize results."""

        embedding = await self._embedding_service.embed_text(
            text=query.query,
        )

        vector_result = await self._vector_store.search(
            query=VectorSearchQuery(
                vector=embedding.vector,
                tenant_id=query.tenant_id,
                top_k=query.top_k,
                required_tags=query.required_tags,
                allowed_classifications=(query.allowed_classifications),
            )
        )

        items = tuple(
            RetrievedKnowledgeChunk(
                chunk=item.record.chunk,
                rank=item.rank,
                score=item.score,
            )
            for item in vector_result.items
        )

        return KnowledgeRetrievalResult(
            query=query,
            items=items,
        )
