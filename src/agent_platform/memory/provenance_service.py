from agent_platform.memory.citation_registry import (
    CitationRegistry,
)
from agent_platform.memory.contracts import RetrievedItem
from agent_platform.memory.knowledge import (
    KnowledgeRetrievalResult,
)
from agent_platform.memory.provenance import (
    CitedEvidence,
    EvidenceProvenance,
    EvidenceSourceType,
)


class EvidenceProvenanceService:
    """Creates stable citation-ready evidence from governed retrieval."""

    def from_enterprise_knowledge(
        self,
        *,
        result: KnowledgeRetrievalResult,
    ) -> CitationRegistry:
        evidence = tuple(
            CitedEvidence(
                citation_id=f"E{index}",
                content=item.chunk.content,
                provenance=EvidenceProvenance(
                    source_type=(EvidenceSourceType.ENTERPRISE_KNOWLEDGE),
                    source_id=item.chunk.chunk_id,
                    document_id=item.chunk.document_id,
                    chunk_id=item.chunk.chunk_id,
                    source_uri=item.chunk.source_uri,
                    document_title=item.chunk.document_title,
                    document_version=item.chunk.document_version,
                    tenant_id=item.chunk.tenant_id,
                    rank=item.rank,
                    score=item.score,
                ),
            )
            for index, item in enumerate(
                result.items,
                start=1,
            )
        )

        return CitationRegistry(
            evidence=evidence,
        )

    def from_long_term_memory(
        self,
        *,
        items: tuple[RetrievedItem, ...],
    ) -> CitationRegistry:
        evidence = tuple(
            CitedEvidence(
                citation_id=f"M{index}",
                content=item.record.content,
                provenance=EvidenceProvenance(
                    source_type=(EvidenceSourceType.LONG_TERM_MEMORY),
                    source_id=item.record.memory_id,
                    memory_id=item.record.memory_id,
                    tenant_id=(item.record.metadata.tenant_id),
                    rank=item.rank,
                    score=item.score,
                ),
            )
            for index, item in enumerate(
                items,
                start=1,
            )
        )

        return CitationRegistry(
            evidence=evidence,
        )
