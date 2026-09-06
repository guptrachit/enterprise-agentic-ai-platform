from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    EnterpriseDocument,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
    KnowledgeRetrievalResult,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
)


def create_document() -> EnterpriseDocument:
    return EnterpriseDocument(
        document_id="doc-001",
        tenant_id="tenant-001",
        title="Customer Support Escalation Policy",
        content="Escalate priority incidents after 24 hours.",
        source_type=KnowledgeSourceType.DOCUMENT,
        source_uri="s3://enterprise/policies/escalation.pdf",
        classification=KnowledgeClassification.INTERNAL,
        version="3.2",
        created_at=datetime(
            2026,
            9,
            4,
            tzinfo=UTC,
        ),
        tags=frozenset(
            {
                "support",
                "policy",
            }
        ),
    )


def create_chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id="doc-001-chunk-000",
        document_id="doc-001",
        tenant_id="tenant-001",
        content="Escalate priority incidents after 24 hours.",
        chunk_index=0,
        source_type=KnowledgeSourceType.DOCUMENT,
        classification=KnowledgeClassification.INTERNAL,
        source_uri="s3://enterprise/policies/escalation.pdf",
        document_title="Customer Support Escalation Policy",
        document_version="3.2",
        tags=frozenset(
            {
                "support",
                "policy",
            }
        ),
    )


def test_enterprise_document_is_immutable() -> None:
    document = create_document()

    with pytest.raises(ValidationError):
        document.title = "Changed"


def test_document_preserves_source_and_security_metadata() -> None:
    document = create_document()

    assert document.document_id == "doc-001"
    assert document.tenant_id == "tenant-001"
    assert document.source_type is KnowledgeSourceType.DOCUMENT
    assert document.classification is KnowledgeClassification.INTERNAL
    assert document.version == "3.2"


def test_chunk_preserves_parent_document_identity() -> None:
    chunk = create_chunk()

    assert chunk.document_id == "doc-001"
    assert chunk.chunk_index == 0
    assert chunk.document_title == "Customer Support Escalation Policy"
    assert chunk.document_version == "3.2"


def test_chunk_requires_non_negative_index() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunk(
            chunk_id="chunk-invalid",
            document_id="doc-001",
            tenant_id="tenant-001",
            content="Invalid",
            chunk_index=-1,
            source_type=KnowledgeSourceType.DOCUMENT,
            classification=KnowledgeClassification.INTERNAL,
        )


def test_knowledge_query_requires_positive_top_k() -> None:
    with pytest.raises(ValidationError):
        KnowledgeQuery(
            query="escalation policy",
            tenant_id="tenant-001",
            top_k=0,
        )


def test_knowledge_query_preserves_security_filters() -> None:
    query = KnowledgeQuery(
        query="incident escalation",
        tenant_id="tenant-001",
        top_k=3,
        allowed_classifications=frozenset(
            {
                KnowledgeClassification.PUBLIC,
                KnowledgeClassification.INTERNAL,
            }
        ),
        required_tags=frozenset(
            {
                "policy",
            }
        ),
    )

    assert query.tenant_id == "tenant-001"
    assert KnowledgeClassification.INTERNAL in query.allowed_classifications
    assert "policy" in query.required_tags


def test_retrieval_result_preserves_rank_and_score() -> None:
    query = KnowledgeQuery(
        query="incident escalation",
        tenant_id="tenant-001",
    )

    chunk = create_chunk()

    retrieved = RetrievedKnowledgeChunk(
        chunk=chunk,
        rank=1,
        score=0.93,
    )

    result = KnowledgeRetrievalResult(
        query=query,
        items=(retrieved,),
    )

    assert result.query == query
    assert result.items[0].chunk == chunk
    assert result.items[0].rank == 1
    assert result.items[0].score == 0.93
