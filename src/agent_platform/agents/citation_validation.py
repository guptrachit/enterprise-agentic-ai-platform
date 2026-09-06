from pydantic import BaseModel, ConfigDict


class CitationValidationResult(BaseModel):
    """Validation of model-produced citation identifiers."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    valid_citation_ids: tuple[str, ...]
    invalid_citation_ids: tuple[str, ...]

    @property
    def is_valid(
        self,
    ) -> bool:
        return not self.invalid_citation_ids


class CitationValidationService:
    """Ensures model citations belong to trusted evidence."""

    def validate(
        self,
        *,
        cited_ids: tuple[str, ...],
        allowed_citation_ids: frozenset[str],
    ) -> CitationValidationResult:
        valid: list[str] = []
        invalid: list[str] = []

        seen: set[str] = set()

        for citation_id in cited_ids:
            if citation_id in seen:
                continue

            seen.add(citation_id)

            if citation_id in allowed_citation_ids:
                valid.append(citation_id)
            else:
                invalid.append(citation_id)

        return CitationValidationResult(
            valid_citation_ids=tuple(valid),
            invalid_citation_ids=tuple(invalid),
        )
