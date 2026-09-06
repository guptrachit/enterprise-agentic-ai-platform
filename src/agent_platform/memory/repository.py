from agent_platform.memory.contracts import (
    MemoryRecord,
    MemoryType,
    RetrievalQuery,
    RetrievalResult,
)
from agent_platform.memory.errors import MemoryRecordNotFoundError
from agent_platform.memory.long_term import LongTermMemoryStore


class MemoryRepository:
    """Platform-owned repository for durable memory operations."""

    def __init__(
        self,
        *,
        store: LongTermMemoryStore,
    ) -> None:
        self._store = store

    async def save(
        self,
        *,
        record: MemoryRecord,
    ) -> MemoryRecord:
        """Persist a valid long-term memory record."""

        if record.metadata.memory_type is not MemoryType.LONG_TERM:
            raise ValueError("MemoryRepository only accepts LONG_TERM memory records")

        return await self._store.save(
            record=record,
        )

    async def get(
        self,
        *,
        memory_id: str,
    ) -> MemoryRecord | None:
        """Return a memory record if it exists."""

        return await self._store.get(
            memory_id=memory_id,
        )

    async def get_required(
        self,
        *,
        memory_id: str,
    ) -> MemoryRecord:
        """Return a memory record or raise a typed not-found error."""

        record = await self.get(
            memory_id=memory_id,
        )

        if record is None:
            raise MemoryRecordNotFoundError(f"Memory record not found: {memory_id}")

        return record

    async def search(
        self,
        *,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        """Search long-term memory through the configured store."""

        return await self._store.search(
            query=query,
        )

    async def delete(
        self,
        *,
        memory_id: str,
    ) -> bool:
        """Delete a durable memory record."""

        return await self._store.delete(
            memory_id=memory_id,
        )
