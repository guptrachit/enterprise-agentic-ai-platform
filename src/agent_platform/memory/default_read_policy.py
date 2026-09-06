from agent_platform.memory.contracts import MemoryScope
from agent_platform.memory.read_policy import (
    MemoryReadContext,
    MemoryReadDecision,
    MemoryReadEvaluation,
)


class DefaultMemoryReadPolicy:
    """Fail-closed policy for governed durable-memory reads."""

    def authorize_query(
        self,
        *,
        query,
        context: MemoryReadContext,
    ) -> MemoryReadEvaluation:
        if query.tenant_id is None:
            return MemoryReadEvaluation(
                decision=MemoryReadDecision.DENY,
                reason="Retrieval query requires tenant_id",
            )

        if query.tenant_id != context.tenant_id:
            return MemoryReadEvaluation(
                decision=MemoryReadDecision.DENY,
                reason="Retrieval query tenant does not match caller tenant",
            )

        if query.subject_id is not None and query.subject_id != context.subject_id:
            return MemoryReadEvaluation(
                decision=MemoryReadDecision.DENY,
                reason="Retrieval query subject does not match caller subject",
            )

        return MemoryReadEvaluation(
            decision=MemoryReadDecision.ALLOW,
            reason="Retrieval query satisfies memory-read policy",
        )

    def authorize_record(
        self,
        *,
        record,
        context: MemoryReadContext,
    ) -> MemoryReadEvaluation:
        metadata = record.metadata

        if metadata.tenant_id is None:
            return MemoryReadEvaluation(
                decision=MemoryReadDecision.DENY,
                reason="Stored memory is missing tenant_id",
            )

        if metadata.tenant_id != context.tenant_id:
            return MemoryReadEvaluation(
                decision=MemoryReadDecision.DENY,
                reason="Stored memory belongs to another tenant",
            )

        if metadata.scope is MemoryScope.USER:
            if context.subject_id is None:
                return MemoryReadEvaluation(
                    decision=MemoryReadDecision.DENY,
                    reason="USER-scoped memory requires caller subject_id",
                )

            if metadata.subject_id != context.subject_id:
                return MemoryReadEvaluation(
                    decision=MemoryReadDecision.DENY,
                    reason="USER-scoped memory belongs to another subject",
                )

        return MemoryReadEvaluation(
            decision=MemoryReadDecision.ALLOW,
            reason="Stored memory satisfies memory-read policy",
        )
