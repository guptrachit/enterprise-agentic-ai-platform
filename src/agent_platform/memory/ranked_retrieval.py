from agent_platform.memory.knowledge import (
    KnowledgeQuery,
    KnowledgeRetrievalResult,
)
from agent_platform.memory.knowledge_authorization import (
    KnowledgeAuthorizationContext,
)
from agent_platform.memory.ranking_service import (
    KnowledgeRankingService,
)
from agent_platform.memory.secure_semantic_retrieval import (
    SecureSemanticRetrievalService,
)


class RankedKnowledgeRetrievalService:
    """Secure semantic retrieval followed by platform-owned reranking."""

    def __init__(
        self,
        *,
        retrieval_service: SecureSemanticRetrievalService,
        ranking_service: KnowledgeRankingService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._ranking_service = ranking_service

    async def retrieve(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeRetrievalResult:
        secured_result = await self._retrieval_service.retrieve(
            query=query,
            context=context,
        )

        return await self._ranking_service.rank(
            result=secured_result,
        )
