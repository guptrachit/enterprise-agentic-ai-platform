import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    CharacterDocumentChunker,
    ChunkingConfiguration,
    DocumentChunker,
    DocumentPreparationService,
    EnterpriseDocument,
    KnowledgeClassification,
    KnowledgeSourceType,
)


def create_document(
    *,
    content: str = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
) -> EnterpriseDocument:
    return EnterpriseDocument(
        document_id="doc-001",
        tenant_id="tenant-001",
        title="Escalation Policy",
        content=content,
        source_type=KnowledgeSourceType.DOCUMENT,
        source_uri="s3://enterprise/policies/escalation.pdf",
        classification=KnowledgeClassification.INTERNAL,
        version="3.2",
        tags=frozenset(
            {
                "policy",
                "support",
            }
        ),
        attributes={
            "department": "operations",
        },
    )


def test_character_chunker_satisfies_protocol() -> None:
    chunker = CharacterDocumentChunker(
        configuration=ChunkingConfiguration(
            chunk_size=20,
            chunk_overlap=5,
        )
    )

    assert isinstance(
        chunker,
        DocumentChunker,
    )


def test_chunking_configuration_requires_positive_chunk_size() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfiguration(
            chunk_size=0,
        )


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="chunk_overlap",
    ):
        ChunkingConfiguration(
            chunk_size=10,
            chunk_overlap=10,
        )


def test_document_is_split_into_multiple_chunks() -> None:
    chunker = CharacterDocumentChunker(
        configuration=ChunkingConfiguration(
            chunk_size=20,
            chunk_overlap=5,
        )
    )

    chunks = chunker.chunk(
        document=create_document(),
    )

    assert len(chunks) > 1


def test_chunk_ids_and_indexes_are_deterministic() -> None:
    chunker = CharacterDocumentChunker(
        configuration=ChunkingConfiguration(
            chunk_size=20,
            chunk_overlap=5,
        )
    )

    chunks = chunker.chunk(
        document=create_document(),
    )

    assert chunks[0].chunk_id == "doc-001-chunk-0000"
    assert chunks[0].chunk_index == 0

    assert chunks[1].chunk_id == "doc-001-chunk-0001"
    assert chunks[1].chunk_index == 1


def test_chunk_preserves_document_provenance() -> None:
    chunker = CharacterDocumentChunker(
        configuration=ChunkingConfiguration(
            chunk_size=20,
            chunk_overlap=5,
        )
    )

    chunk = chunker.chunk(
        document=create_document(),
    )[0]

    assert chunk.document_id == "doc-001"
    assert chunk.tenant_id == "tenant-001"
    assert chunk.document_title == "Escalation Policy"
    assert chunk.document_version == "3.2"
    assert chunk.source_uri == "s3://enterprise/policies/escalation.pdf"
    assert chunk.classification is KnowledgeClassification.INTERNAL


def test_chunk_preserves_document_attributes_and_offsets() -> None:
    chunker = CharacterDocumentChunker(
        configuration=ChunkingConfiguration(
            chunk_size=20,
            chunk_overlap=5,
        )
    )

    chunk = chunker.chunk(
        document=create_document(),
    )[0]

    assert chunk.attributes["department"] == "operations"
    assert chunk.attributes["start_char"] == 0
    assert chunk.attributes["end_char"] == 20


def test_overlap_is_preserved_between_chunks() -> None:
    document = create_document(
        content="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )

    chunker = CharacterDocumentChunker(
        configuration=ChunkingConfiguration(
            chunk_size=10,
            chunk_overlap=2,
        )
    )

    chunks = chunker.chunk(
        document=document,
    )

    assert chunks[0].content == "ABCDEFGHIJ"
    assert chunks[1].content == "IJKLMNOPQR"


def test_short_document_produces_single_chunk() -> None:
    chunker = CharacterDocumentChunker(
        configuration=ChunkingConfiguration(
            chunk_size=100,
            chunk_overlap=10,
        )
    )

    chunks = chunker.chunk(
        document=create_document(
            content="Short document.",
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Short document."


def test_document_preparation_service_delegates_to_chunker() -> None:
    service = DocumentPreparationService(
        chunker=CharacterDocumentChunker(
            configuration=ChunkingConfiguration(
                chunk_size=20,
                chunk_overlap=5,
            )
        )
    )

    chunks = service.prepare(
        document=create_document(),
    )

    assert len(chunks) > 1
    assert chunks[0].document_id == "doc-001"
