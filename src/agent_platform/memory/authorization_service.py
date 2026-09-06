from agent_platform.memory.authorization import (
    MemoryAuthorizationContext,
    MemoryAuthorizationDecision,
    MemoryAuthorizationDeniedError,
    MemoryAuthorizationPolicy,
    MemoryOperation,
)
from agent_platform.memory.contracts import MemoryRecord


class MemoryAuthorizationService:
    """Centralized fail-closed memory authorization service."""

    def __init__(
        self,
        *,
        policy: MemoryAuthorizationPolicy,
    ) -> None:
        self._policy = policy

    def authorize(
        self,
        *,
        operation: MemoryOperation,
        record: MemoryRecord,
        context: MemoryAuthorizationContext,
    ) -> None:
        """Authorize an operation or raise a typed denial error."""

        evaluation = self._policy.authorize_record(
            operation=operation,
            record=record,
            context=context,
        )

        if evaluation.decision is MemoryAuthorizationDecision.DENY:
            raise MemoryAuthorizationDeniedError(evaluation.reason)
