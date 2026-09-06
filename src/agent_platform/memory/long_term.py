from typing import Protocol, runtime_checkable

from agent_platform.memory.contracts import (
    MemoryRecord,
    RetrievalQuery,
    RetrievalResult,
)


@runtime_checkable
class LongTermMemoryStore(Protocol):
    """Provider-neutral contract for durable memory persistence."""

    async def save(
        self,
        *,
        record: MemoryRecord,
    ) -> MemoryRecord:
        """Persist or update a governed long-term memory record."""
        ...

    async def get(
        self,
        *,
        memory_id: str,
    ) -> MemoryRecord | None:
        """Retrieve a memory record by exact identifier."""
        ...

    async def search(
        self,
        *,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        """Retrieve governed memory matching the supplied query."""
        ...

    async def delete(
        self,
        *,
        memory_id: str,
    ) -> bool:
        """Delete a memory record and return whether it existed."""
        ...
