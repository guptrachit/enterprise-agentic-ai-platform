from agent_platform.memory.knowledge import (
    KnowledgeRetrievalResult,
    RetrievedKnowledgeChunk,
)
from agent_platform.memory.reranking import (
    Reranker,
    RerankRequest,
)


class KnowledgeRankingService:
    """Platform-owned orchestration for final evidence ranking."""

    def __init__(
        self,
        *,
        reranker: Reranker,
    ) -> None:
        self._reranker = reranker

    async def rank(
        self,
        *,
        result: KnowledgeRetrievalResult,
        top_k: int | None = None,
    ) -> KnowledgeRetrievalResult:
        """Rerank retrieved evidence and return normalized results."""

        if not result.items:
            return result

        effective_top_k = top_k if top_k is not None else result.query.top_k

        reranked = await self._reranker.rerank(
            request=RerankRequest(
                query=result.query,
                candidates=result.items,
                top_k=effective_top_k,
            )
        )

        items = tuple(
            RetrievedKnowledgeChunk(
                chunk=item.item.chunk,
                score=item.rerank_score,
                rank=item.rank,
            )
            for item in reranked.items
        )

        return KnowledgeRetrievalResult(
            query=result.query,
            items=items,
        )
