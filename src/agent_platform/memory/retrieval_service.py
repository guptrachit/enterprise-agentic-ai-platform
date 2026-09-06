from agent_platform.memory.contracts import (
    RetrievalQuery,
    RetrievalResult,
)
from agent_platform.memory.read_policy import (
    MemoryReadContext,
    MemoryReadDecision,
    MemoryReadDeniedError,
    MemoryReadPolicy,
)
from agent_platform.memory.repository import MemoryRepository


class MemoryRetrievalService:
    """Governed entry point for durable memory retrieval."""

    def __init__(
        self,
        *,
        repository: MemoryRepository,
        policy: MemoryReadPolicy,
    ) -> None:
        self._repository = repository
        self._policy = policy

    async def search(
        self,
        *,
        query: RetrievalQuery,
        context: MemoryReadContext,
    ) -> RetrievalResult:
        """Authorize query, retrieve records, and filter every result."""

        query_evaluation = self._policy.authorize_query(
            query=query,
            context=context,
        )

        if query_evaluation.decision is MemoryReadDecision.DENY:
            raise MemoryReadDeniedError(query_evaluation.reason)

        result = await self._repository.search(
            query=query,
        )

        authorized_items = tuple(
            item
            for item in result.items
            if self._policy.authorize_record(
                record=item.record,
                context=context,
            ).decision
            is MemoryReadDecision.ALLOW
        )

        return RetrievalResult(
            query=result.query,
            items=authorized_items,
        )
