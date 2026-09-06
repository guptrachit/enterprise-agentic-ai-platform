from agent_platform.memory import (
    CitationRegistry,
    EvidenceProvenanceService,
    EvidenceSourceType,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
    KnowledgeRetrievalResult,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    build_citation_source_map,
)


def create_result() -> KnowledgeRetrievalResult:
    query = KnowledgeQuery(
        query="incident escalation",
        tenant_id="tenant-001",
        top_k=2,
    )

    items = (
        RetrievedKnowledgeChunk(
            chunk=KnowledgeChunk(
                chunk_id="chunk-001",
                document_id="doc-001",
                tenant_id="tenant-001",
                content="Escalation policy one.",
                chunk_index=0,
                source_type=KnowledgeSourceType.DOCUMENT,
                classification=(KnowledgeClassification.INTERNAL),
                source_uri="s3://bucket/doc-001.pdf",
                document_title="Incident Policy",
                document_version="v3",
            ),
            rank=1,
            score=0.95,
        ),
        RetrievedKnowledgeChunk(
            chunk=KnowledgeChunk(
                chunk_id="chunk-002",
                document_id="doc-002",
                tenant_id="tenant-001",
                content="Escalation policy two.",
                chunk_index=0,
                source_type=KnowledgeSourceType.DOCUMENT,
                classification=(KnowledgeClassification.INTERNAL),
                source_uri="s3://bucket/doc-002.pdf",
                document_title="Escalation Manual",
                document_version="v1",
            ),
            rank=2,
            score=0.81,
        ),
    )

    return KnowledgeRetrievalResult(
        query=query,
        items=items,
    )


def test_enterprise_evidence_gets_stable_citation_ids() -> None:
    service = EvidenceProvenanceService()

    registry = service.from_enterprise_knowledge(
        result=create_result(),
    )

    assert registry.citation_ids == (
        "E1",
        "E2",
    )


def test_enterprise_provenance_preserves_document_metadata() -> None:
    service = EvidenceProvenanceService()

    registry = service.from_enterprise_knowledge(
        result=create_result(),
    )

    evidence = registry.get("E1")

    assert evidence is not None

    assert evidence.provenance.source_type is EvidenceSourceType.ENTERPRISE_KNOWLEDGE

    assert evidence.provenance.document_id == "doc-001"
    assert evidence.provenance.chunk_id == "chunk-001"

    assert evidence.provenance.document_title == "Incident Policy"

    assert evidence.provenance.document_version == "v3"

    assert evidence.provenance.source_uri == "s3://bucket/doc-001.pdf"


def test_provenance_preserves_rank_and_score() -> None:
    service = EvidenceProvenanceService()

    registry = service.from_enterprise_knowledge(
        result=create_result(),
    )

    evidence = registry.get("E1")

    assert evidence is not None
    assert evidence.provenance.rank == 1
    assert evidence.provenance.score == 0.95


def test_registry_get_returns_none_for_unknown_citation() -> None:
    registry = CitationRegistry()

    assert registry.get("E99") is None
    assert registry.contains("E99") is False


def test_source_map_links_chunk_to_citation() -> None:
    service = EvidenceProvenanceService()

    registry = service.from_enterprise_knowledge(
        result=create_result(),
    )

    mapping = build_citation_source_map(
        registry=registry,
    )

    assert mapping == {
        "chunk-001": "E1",
        "chunk-002": "E2",
    }
