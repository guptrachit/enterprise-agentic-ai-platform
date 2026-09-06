from agent_platform.memory.contracts import (
    MemoryScope,
    MemoryType,
)
from agent_platform.memory.write_policy import (
    MemoryWriteDecision,
    MemoryWriteEvaluation,
    MemoryWriteRequest,
)


class DefaultMemoryWritePolicy:
    """Conservative fail-closed policy for long-term memory persistence."""

    def evaluate(
        self,
        *,
        request: MemoryWriteRequest,
    ) -> MemoryWriteEvaluation:
        metadata = request.metadata

        if metadata.memory_type is not MemoryType.LONG_TERM:
            return MemoryWriteEvaluation(
                decision=MemoryWriteDecision.DENY,
                reason="Only LONG_TERM memory may be durably persisted",
            )

        if metadata.tenant_id is None:
            return MemoryWriteEvaluation(
                decision=MemoryWriteDecision.DENY,
                reason="Durable memory requires tenant_id",
            )

        if metadata.scope is MemoryScope.USER and metadata.subject_id is None:
            return MemoryWriteEvaluation(
                decision=MemoryWriteDecision.DENY,
                reason="USER-scoped memory requires subject_id",
            )

        return MemoryWriteEvaluation(
            decision=MemoryWriteDecision.ALLOW,
            reason="Memory candidate satisfies durable write policy",
        )
