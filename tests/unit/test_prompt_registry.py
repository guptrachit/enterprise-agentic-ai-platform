import pytest

from agent_platform.llm.errors import (
    LLMPromptActiveVersionNotSetError,
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


def test_prompt_registry_lists_versions() -> None:
    registry = PromptRegistry()

    registry.register(
        PromptTemplate(
            name="ticket_classifier",
            version="2.0",
            template="Version 2: {ticket_text}",
        )
    )

    registry.register(
        PromptTemplate(
            name="ticket_classifier",
            version="1.0",
            template="Version 1: {ticket_text}",
        )
    )

    versions = registry.list_versions(
        "ticket_classifier",
    )

    assert versions == (
        "1.0",
        "2.0",
    )


def test_prompt_registry_returns_active_version() -> None:
    registry = PromptRegistry()

    version_1 = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Version 1: {ticket_text}",
    )

    version_2 = PromptTemplate(
        name="ticket_classifier",
        version="2.0",
        template="Version 2: {ticket_text}",
    )

    registry.register(version_1)
    registry.register(version_2)

    registry.set_active(
        "ticket_classifier",
        "2.0",
    )

    result = registry.get_active(
        "ticket_classifier",
    )

    assert result is version_2


def test_prompt_registry_can_switch_active_version() -> None:
    registry = PromptRegistry()

    version_1 = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Version 1: {ticket_text}",
    )

    version_2 = PromptTemplate(
        name="ticket_classifier",
        version="2.0",
        template="Version 2: {ticket_text}",
    )

    registry.register(version_1)
    registry.register(version_2)

    registry.set_active(
        "ticket_classifier",
        "1.0",
    )

    assert registry.get_active("ticket_classifier") is version_1

    registry.set_active(
        "ticket_classifier",
        "2.0",
    )

    assert registry.get_active("ticket_classifier") is version_2


def test_prompt_registry_rejects_unknown_active_version() -> None:
    registry = PromptRegistry()

    registry.register(
        PromptTemplate(
            name="ticket_classifier",
            version="1.0",
            template="Version 1: {ticket_text}",
        )
    )

    with pytest.raises(LLMPromptNotFoundError):
        registry.set_active(
            "ticket_classifier",
            "9.9",
        )


def test_prompt_registry_raises_when_active_version_not_set() -> None:
    registry = PromptRegistry()

    registry.register(
        PromptTemplate(
            name="ticket_classifier",
            version="1.0",
            template="Version 1: {ticket_text}",
        )
    )

    with pytest.raises(
        LLMPromptActiveVersionNotSetError,
        match="ticket_classifier",
    ):
        registry.get_active(
            "ticket_classifier",
        )
