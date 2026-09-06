from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceSourceType(StrEnum):
    CONVERSATION = "conversation"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    ENTERPRISE_KNOWLEDGE = "enterprise_knowledge"


class EvidenceProvenance(BaseModel):
    """Traceable source metadata for one context/evidence item."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    source_type: EvidenceSourceType

    source_id: str = Field(
        min_length=1,
    )

    document_id: str | None = None
    chunk_id: str | None = None
    memory_id: str | None = None

    source_uri: str | None = None
    document_title: str | None = None
    document_version: str | None = None

    tenant_id: str | None = None

    rank: int | None = Field(
        default=None,
        ge=1,
    )

    score: float | None = None


class CitedEvidence(BaseModel):
    """One evidence item plus provenance available for citation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    citation_id: str = Field(
        min_length=1,
    )

    content: str = Field(
        min_length=1,
    )

    provenance: EvidenceProvenance
