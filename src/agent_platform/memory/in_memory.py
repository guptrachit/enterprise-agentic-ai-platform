from agent_platform.memory.contracts import (
    MemoryRecord,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
)


class InMemoryLongTermMemoryStore:
    """Deterministic in-memory implementation of durable memory storage."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    async def save(
        self,
        *,
        record: MemoryRecord,
    ) -> MemoryRecord:
        self._records[record.memory_id] = record
        return record

    async def get(
        self,
        *,
        memory_id: str,
    ) -> MemoryRecord | None:
        return self._records.get(memory_id)

    async def search(
        self,
        *,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        matching_records = tuple(
            record
            for record in self._records.values()
            if self._matches_query(
                record=record,
                query=query,
            )
        )

        items = tuple(
            RetrievedItem(
                record=record,
                rank=rank,
            )
            for rank, record in enumerate(
                matching_records[: query.top_k],
                start=1,
            )
        )

        return RetrievalResult(
            query=query,
            items=items,
        )

    async def delete(
        self,
        *,
        memory_id: str,
    ) -> bool:
        return (
            self._records.pop(
                memory_id,
                None,
            )
            is not None
        )

    @staticmethod
    def _matches_query(
        *,
        record: MemoryRecord,
        query: RetrievalQuery,
    ) -> bool:
        metadata = record.metadata

        if query.subject_id is not None and metadata.subject_id != query.subject_id:
            return False

        if query.tenant_id is not None and metadata.tenant_id != query.tenant_id:
            return False

        if (
            query.conversation_id is not None
            and metadata.conversation_id != query.conversation_id
        ):
            return False

        if not query.required_tags.issubset(
            metadata.tags,
        ):
            return False

        normalized_query = query.query.casefold()
        normalized_content = record.content.casefold()

        return normalized_query in normalized_content
