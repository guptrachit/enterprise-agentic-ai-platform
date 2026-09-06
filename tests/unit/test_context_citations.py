from agent_platform.memory import (
    ContextAssemblyInput,
    ContextAssemblyService,
    ContextBudget,
    DeterministicTokenEstimator,
    EvidenceProvenanceService,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
    KnowledgeRetrievalResult,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    build_citation_source_map,
)


def create_retrieval_result() -> KnowledgeRetrievalResult:
    return KnowledgeRetrievalResult(
        query=KnowledgeQuery(
            query="incident policy",
            tenant_id="tenant-001",
        ),
        items=(
            RetrievedKnowledgeChunk(
                chunk=KnowledgeChunk(
                    chunk_id="chunk-001",
                    document_id="doc-001",
                    tenant_id="tenant-001",
                    content="Critical incidents require escalation.",
                    chunk_index=0,
                    source_type=KnowledgeSourceType.DOCUMENT,
                    classification=(KnowledgeClassification.INTERNAL),
                ),
                rank=1,
                score=0.9,
            ),
        ),
    )


def test_context_fragment_keeps_citation_id() -> None:
    retrieval = create_retrieval_result()

    provenance_service = EvidenceProvenanceService()

    registry = provenance_service.from_enterprise_knowledge(
        result=retrieval,
    )

    service = ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))

    result = service.assemble(
        input_data=ContextAssemblyInput(
            enterprise_knowledge=retrieval,
            citation_ids_by_source=(
                build_citation_source_map(
                    registry=registry,
                )
            ),
        ),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert len(result.fragments) == 1
    assert result.fragments[0].citation_id == "E1"
    assert result.fragments[0].source_id == "chunk-001"
