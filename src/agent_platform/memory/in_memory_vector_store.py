from math import sqrt

from agent_platform.memory.embeddings import EmbeddingVector
from agent_platform.memory.vector_store import (
    VectorRecord,
    VectorSearchQuery,
    VectorSearchResult,
    VectorSearchResultItem,
)


class InMemoryVectorStore:
    """Deterministic in-memory vector store using cosine similarity."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(
        self,
        *,
        record: VectorRecord,
    ) -> VectorRecord:
        self._records[record.vector_id] = record
        return record

    async def get(
        self,
        *,
        vector_id: str,
    ) -> VectorRecord | None:
        return self._records.get(vector_id)

    async def search(
        self,
        *,
        query: VectorSearchQuery,
    ) -> VectorSearchResult:
        scored_records: list[tuple[VectorRecord, float]] = []

        for record in self._records.values():
            if record.chunk.tenant_id != query.tenant_id:
                continue

            if not query.required_tags.issubset(
                record.chunk.tags,
            ):
                continue

            if (
                query.allowed_classifications
                and record.chunk.classification not in query.allowed_classifications
            ):
                continue

            score = self._cosine_similarity(
                left=query.vector,
                right=record.vector,
            )

            scored_records.append(
                (
                    record,
                    score,
                )
            )

        scored_records.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        items = tuple(
            VectorSearchResultItem(
                record=record,
                score=score,
                rank=rank,
            )
            for rank, (record, score) in enumerate(
                scored_records[: query.top_k],
                start=1,
            )
        )

        return VectorSearchResult(
            query=query,
            items=items,
        )

    async def delete(
        self,
        *,
        vector_id: str,
    ) -> bool:
        return (
            self._records.pop(
                vector_id,
                None,
            )
            is not None
        )

    @staticmethod
    def _cosine_similarity(
        *,
        left: EmbeddingVector,
        right: EmbeddingVector,
    ) -> float:
        if left.dimensions != right.dimensions:
            raise ValueError("Embedding dimensions must match")

        dot_product = sum(
            left_value * right_value
            for left_value, right_value in zip(
                left.values,
                right.values,
                strict=True,
            )
        )

        left_magnitude = sqrt(sum(value * value for value in left.values))

        right_magnitude = sqrt(sum(value * value for value in right.values))

        if left_magnitude == 0 or right_magnitude == 0:
            return 0.0

        return dot_product / (left_magnitude * right_magnitude)
