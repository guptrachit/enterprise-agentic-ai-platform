from agent_platform.memory.knowledge import (
    KnowledgeQuery,
    KnowledgeRetrievalResult,
)
from agent_platform.memory.knowledge_authorization import (
    KnowledgeAccessDecision,
    KnowledgeAccessDeniedError,
    KnowledgeAuthorizationContext,
    KnowledgeAuthorizationPolicy,
)
from agent_platform.memory.semantic_retrieval import (
    SemanticRetrievalService,
)


class SecureSemanticRetrievalService:
    """Security-aware facade over semantic enterprise retrieval."""

    def __init__(
        self,
        *,
        retrieval_service: SemanticRetrievalService,
        authorization_policy: KnowledgeAuthorizationPolicy,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._authorization_policy = authorization_policy

    async def retrieve(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeRetrievalResult:
        query_evaluation = self._authorization_policy.authorize_query(
            query=query,
            context=context,
        )

        if query_evaluation.decision is KnowledgeAccessDecision.DENY:
            raise KnowledgeAccessDeniedError(query_evaluation.reason)

        effective_query = self._build_effective_query(
            query=query,
            context=context,
        )

        result = await self._retrieval_service.retrieve(
            query=effective_query,
        )

        authorized_items = tuple(
            item
            for item in result.items
            if (
                self._authorization_policy.authorize_chunk(
                    chunk=item.chunk,
                    context=context,
                ).decision
                is KnowledgeAccessDecision.ALLOW
            )
        )

        return KnowledgeRetrievalResult(
            query=query,
            items=authorized_items,
        )

    @staticmethod
    def _build_effective_query(
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeQuery:
        if query.allowed_classifications:
            classifications = query.allowed_classifications
        else:
            classifications = context.allowed_classifications

        return query.model_copy(
            update={
                "allowed_classifications": classifications,
            }
        )
