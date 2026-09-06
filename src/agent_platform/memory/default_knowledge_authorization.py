from agent_platform.memory.knowledge import (
    KnowledgeChunk,
    KnowledgeQuery,
)
from agent_platform.memory.knowledge_authorization import (
    KnowledgeAccessDecision,
    KnowledgeAccessEvaluation,
    KnowledgeAuthorizationContext,
)


class DefaultKnowledgeAuthorizationPolicy:
    """Fail-closed policy for enterprise knowledge retrieval."""

    _REQUIRED_SCOPE = "knowledge:read"

    def authorize_query(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeAccessEvaluation:
        if self._REQUIRED_SCOPE not in context.granted_scopes:
            return KnowledgeAccessEvaluation(
                decision=KnowledgeAccessDecision.DENY,
                reason="Missing required scope: knowledge:read",
            )

        if query.tenant_id != context.tenant_id:
            return KnowledgeAccessEvaluation(
                decision=KnowledgeAccessDecision.DENY,
                reason="Knowledge query targets another tenant",
            )

        if query.allowed_classifications and not query.allowed_classifications.issubset(
            context.allowed_classifications
        ):
            return KnowledgeAccessEvaluation(
                decision=KnowledgeAccessDecision.DENY,
                reason=(
                    "Knowledge query requests classifications "
                    "outside caller authorization"
                ),
            )

        return KnowledgeAccessEvaluation(
            decision=KnowledgeAccessDecision.ALLOW,
            reason="Knowledge query satisfies authorization policy",
        )

    def authorize_chunk(
        self,
        *,
        chunk: KnowledgeChunk,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeAccessEvaluation:
        if self._REQUIRED_SCOPE not in context.granted_scopes:
            return KnowledgeAccessEvaluation(
                decision=KnowledgeAccessDecision.DENY,
                reason="Missing required scope: knowledge:read",
            )

        if chunk.tenant_id != context.tenant_id:
            return KnowledgeAccessEvaluation(
                decision=KnowledgeAccessDecision.DENY,
                reason="Knowledge chunk belongs to another tenant",
            )

        if chunk.classification not in context.allowed_classifications:
            return KnowledgeAccessEvaluation(
                decision=KnowledgeAccessDecision.DENY,
                reason=("Knowledge classification is not authorized for this caller"),
            )

        return KnowledgeAccessEvaluation(
            decision=KnowledgeAccessDecision.ALLOW,
            reason="Knowledge chunk satisfies authorization policy",
        )
