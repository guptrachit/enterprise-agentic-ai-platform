from typing import ClassVar

from agent_platform.memory.authorization import (
    MemoryAuthorizationContext,
    MemoryAuthorizationDecision,
    MemoryAuthorizationEvaluation,
    MemoryOperation,
)
from agent_platform.memory.contracts import (
    MemoryRecord,
    MemoryScope,
)


class DefaultMemoryAuthorizationPolicy:
    """Fail-closed enterprise authorization policy for memory."""

    _REQUIRED_SCOPES: ClassVar[dict[MemoryOperation, str]] = {
        MemoryOperation.READ: "memory:read",
        MemoryOperation.WRITE: "memory:write",
        MemoryOperation.DELETE: "memory:delete",
    }

    def authorize_record(
        self,
        *,
        operation: MemoryOperation,
        record: MemoryRecord,
        context: MemoryAuthorizationContext,
    ) -> MemoryAuthorizationEvaluation:
        required_scope = self._REQUIRED_SCOPES[operation]

        if required_scope not in context.granted_scopes:
            return MemoryAuthorizationEvaluation(
                decision=MemoryAuthorizationDecision.DENY,
                reason=f"Missing required scope: {required_scope}",
            )

        metadata = record.metadata

        if metadata.tenant_id is None:
            return MemoryAuthorizationEvaluation(
                decision=MemoryAuthorizationDecision.DENY,
                reason="Memory record is missing tenant_id",
            )

        if metadata.tenant_id != context.tenant_id:
            return MemoryAuthorizationEvaluation(
                decision=MemoryAuthorizationDecision.DENY,
                reason="Memory record belongs to another tenant",
            )

        if metadata.scope is MemoryScope.USER:
            if context.subject_id is None:
                return MemoryAuthorizationEvaluation(
                    decision=MemoryAuthorizationDecision.DENY,
                    reason="USER-scoped memory requires caller subject_id",
                )

            if metadata.subject_id != context.subject_id:
                return MemoryAuthorizationEvaluation(
                    decision=MemoryAuthorizationDecision.DENY,
                    reason="Memory record belongs to another subject",
                )

        return MemoryAuthorizationEvaluation(
            decision=MemoryAuthorizationDecision.ALLOW,
            reason="Memory operation satisfies authorization policy",
        )
