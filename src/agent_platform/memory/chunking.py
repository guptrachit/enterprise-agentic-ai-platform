from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.knowledge import (
    EnterpriseDocument,
    KnowledgeChunk,
)


class ChunkingConfiguration(BaseModel):
    """Deterministic configuration for enterprise document chunking."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)

    def model_post_init(
        self,
        __context,
        /,
    ) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@runtime_checkable
class DocumentChunker(Protocol):
    """Provider-neutral contract for document chunking."""

    def chunk(
        self,
        *,
        document: EnterpriseDocument,
    ) -> tuple[KnowledgeChunk, ...]:
        """Convert an enterprise document into retrieval chunks."""
        ...


class CharacterDocumentChunker:
    """Deterministic character-based document chunker."""

    def __init__(
        self,
        *,
        configuration: ChunkingConfiguration,
    ) -> None:
        self._configuration = configuration

    def chunk(
        self,
        *,
        document: EnterpriseDocument,
    ) -> tuple[KnowledgeChunk, ...]:
        content = document.content.strip()

        if not content:
            return ()

        chunk_size = self._configuration.chunk_size
        chunk_overlap = self._configuration.chunk_overlap

        step = chunk_size - chunk_overlap

        chunks: list[KnowledgeChunk] = []

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(
                start + chunk_size,
                len(content),
            )

            chunk_content = content[start:end].strip()

            if chunk_content:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=(f"{document.document_id}-chunk-{chunk_index:04d}"),
                        document_id=document.document_id,
                        tenant_id=document.tenant_id,
                        content=chunk_content,
                        chunk_index=chunk_index,
                        source_type=document.source_type,
                        classification=document.classification,
                        source_uri=document.source_uri,
                        document_title=document.title,
                        document_version=document.version,
                        tags=document.tags,
                        attributes={
                            **document.attributes,
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )

                chunk_index += 1

            if end == len(content):
                break

            start += step

        return tuple(chunks)
