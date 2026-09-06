from agent_platform.memory.authorization import (
    MemoryAuthorizationContext,
    MemoryOperation,
)
from agent_platform.memory.authorization_service import (
    MemoryAuthorizationService,
)
from agent_platform.memory.contracts import MemoryRecord
from agent_platform.memory.repository import MemoryRepository


class SecuredMemoryRepository:
    """Authorization-enforcing facade over durable memory repository."""

    def __init__(
        self,
        *,
        repository: MemoryRepository,
        authorization_service: MemoryAuthorizationService,
    ) -> None:
        self._repository = repository
        self._authorization_service = authorization_service

    async def save(
        self,
        *,
        record: MemoryRecord,
        context: MemoryAuthorizationContext,
    ) -> MemoryRecord:
        self._authorization_service.authorize(
            operation=MemoryOperation.WRITE,
            record=record,
            context=context,
        )

        return await self._repository.save(
            record=record,
        )

    async def get(
        self,
        *,
        memory_id: str,
        context: MemoryAuthorizationContext,
    ) -> MemoryRecord | None:
        record = await self._repository.get(
            memory_id=memory_id,
        )

        if record is None:
            return None

        self._authorization_service.authorize(
            operation=MemoryOperation.READ,
            record=record,
            context=context,
        )

        return record

    async def delete(
        self,
        *,
        memory_id: str,
        context: MemoryAuthorizationContext,
    ) -> bool:
        record = await self._repository.get(
            memory_id=memory_id,
        )

        if record is None:
            return False

        self._authorization_service.authorize(
            operation=MemoryOperation.DELETE,
            record=record,
            context=context,
        )

        return await self._repository.delete(
            memory_id=memory_id,
        )
