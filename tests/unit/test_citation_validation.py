from agent_platform.agents.citation_validation import (
    CitationValidationService,
)


def test_valid_citations_are_accepted() -> None:
    service = CitationValidationService()

    result = service.validate(
        cited_ids=(
            "E1",
            "E2",
        ),
        allowed_citation_ids=frozenset(
            {
                "E1",
                "E2",
            }
        ),
    )

    assert result.is_valid is True

    assert result.valid_citation_ids == (
        "E1",
        "E2",
    )

    assert result.invalid_citation_ids == ()


def test_invented_citation_is_rejected() -> None:
    service = CitationValidationService()

    result = service.validate(
        cited_ids=(
            "E1",
            "E99",
        ),
        allowed_citation_ids=frozenset(
            {
                "E1",
            }
        ),
    )

    assert result.is_valid is False

    assert result.valid_citation_ids == ("E1",)

    assert result.invalid_citation_ids == ("E99",)


def test_duplicate_citations_are_deduplicated() -> None:
    service = CitationValidationService()

    result = service.validate(
        cited_ids=(
            "E1",
            "E1",
            "E1",
        ),
        allowed_citation_ids=frozenset(
            {
                "E1",
            }
        ),
    )

    assert result.valid_citation_ids == ("E1",)


def test_empty_citations_are_valid() -> None:
    service = CitationValidationService()

    result = service.validate(
        cited_ids=(),
        allowed_citation_ids=frozenset(
            {
                "E1",
            }
        ),
    )

    assert result.is_valid is True
