from pydantic import BaseModel, ConfigDict

from agent_platform.memory.provenance import (
    CitedEvidence,
)


class CitationRegistry(BaseModel):
    """Immutable registry of evidence available to the model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    evidence: tuple[CitedEvidence, ...] = ()

    def get(
        self,
        citation_id: str,
    ) -> CitedEvidence | None:
        for item in self.evidence:
            if item.citation_id == citation_id:
                return item

        return None

    def contains(
        self,
        citation_id: str,
    ) -> bool:
        return self.get(citation_id) is not None

    @property
    def citation_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(item.citation_id for item in self.evidence)
