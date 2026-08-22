import pytest

from agent_platform.llm.errors import (
    LLMPromptAlreadyExistsError,
    LLMPromptNotFoundError,
)
from agent_platform.llm.prompt import PromptTemplate
from agent_platform.llm.prompt_registry import PromptRegistry


def test_prompt_registry_registers_and_returns_prompt() -> None:
    registry = PromptRegistry()

    prompt = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Classify this ticket: {ticket_text}",
    )

    registry.register(prompt)

    result = registry.get(
        "ticket_classifier",
        "1.0",
    )

    assert result is prompt


def test_prompt_registry_supports_multiple_versions() -> None:
    registry = PromptRegistry()

    version_1 = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Classify this ticket: {ticket_text}",
    )

    version_2 = PromptTemplate(
        name="ticket_classifier",
        version="2.0",
        template=("Classify this ticket carefully: {ticket_text}"),
    )

    registry.register(version_1)
    registry.register(version_2)

    assert registry.get("ticket_classifier", "1.0") is version_1
    assert registry.get("ticket_classifier", "2.0") is version_2


def test_prompt_registry_rejects_duplicate_version() -> None:
    registry = PromptRegistry()

    prompt = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Classify this ticket: {ticket_text}",
    )

    registry.register(prompt)

    with pytest.raises(
        LLMPromptAlreadyExistsError,
        match="ticket_classifier",
    ):
        registry.register(prompt)


def test_prompt_registry_raises_when_prompt_not_found() -> None:
    registry = PromptRegistry()

    with pytest.raises(
        LLMPromptNotFoundError,
        match="ticket_classifier",
    ):
        registry.get(
            "ticket_classifier",
            "9.9",
        )
