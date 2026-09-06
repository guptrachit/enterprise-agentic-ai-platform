from agent_platform.memory.citation_registry import (
    CitationRegistry,
)


def build_citation_source_map(
    *,
    registry: CitationRegistry,
) -> dict[str, str]:
    """Map source identifiers to stable citation identifiers."""

    return {item.provenance.source_id: item.citation_id for item in registry.evidence}
