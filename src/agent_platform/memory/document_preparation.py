from agent_platform.memory.chunking import DocumentChunker
from agent_platform.memory.knowledge import (
    EnterpriseDocument,
    KnowledgeChunk,
)


class DocumentPreparationService:
    """Platform-owned preparation boundary for enterprise knowledge."""

    def __init__(
        self,
        *,
        chunker: DocumentChunker,
    ) -> None:
        self._chunker = chunker

    def prepare(
        self,
        *,
        document: EnterpriseDocument,
    ) -> tuple[KnowledgeChunk, ...]:
        """Prepare an enterprise document for downstream retrieval."""

        return self._chunker.chunk(
            document=document,
        )
