from agent_platform.agents.citation_validation import (
    CitationValidationService,
)
from agent_platform.memory import (
    ContextAssemblyInput,
    ContextAssemblyService,
    ContextBudget,
    DeterministicTokenEstimator,
)


def test_untrusted_citation_cannot_become_trusted() -> None:
    service = CitationValidationService()

    result = service.validate(
        cited_ids=(
            "E1",
            "E999",
        ),
        allowed_citation_ids=frozenset(
            {
                "E1",
            }
        ),
    )

    assert result.valid_citation_ids == ("E1",)

    assert result.invalid_citation_ids == ("E999",)


def test_context_budget_cannot_be_exceeded() -> None:
    service = ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))

    result = service.assemble(
        input_data=ContextAssemblyInput(),
        budget=ContextBudget(
            max_tokens=1,
        ),
    )

    assert result.used_tokens <= 1


def test_empty_context_is_safe() -> None:
    service = ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))

    result = service.assemble(
        input_data=ContextAssemblyInput(),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert result.fragments == ()
    assert result.used_tokens == 0
    assert result.truncated is False
