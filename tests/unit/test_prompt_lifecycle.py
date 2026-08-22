import pytest

from agent_platform.llm.errors import LLMPromptActiveDeprecationError
from agent_platform.llm.prompt import PromptTemplate
from agent_platform.llm.prompt_lifecycle import PromptStatus
from agent_platform.llm.prompt_registry import PromptRegistry


def make_prompt(version: str) -> PromptTemplate:
    return PromptTemplate(
        name="ticket_classifier",
        version=version,
        template="Classify: {ticket_text}",
    )


def test_registered_prompt_starts_as_draft() -> None:
    registry = PromptRegistry()

    registry.register(
        make_prompt("1.0"),
    )

    assert (
        registry.get_status(
            "ticket_classifier",
            "1.0",
        )
        == PromptStatus.DRAFT
    )


def test_activating_prompt_changes_status_to_active() -> None:
    registry = PromptRegistry()

    registry.register(
        make_prompt("1.0"),
    )

    registry.set_active(
        "ticket_classifier",
        "1.0",
    )

    assert (
        registry.get_status(
            "ticket_classifier",
            "1.0",
        )
        == PromptStatus.ACTIVE
    )


def test_promoting_new_version_deprecates_previous_active() -> None:
    registry = PromptRegistry()

    registry.register(
        make_prompt("1.0"),
    )
    registry.register(
        make_prompt("2.0"),
    )

    registry.set_active(
        "ticket_classifier",
        "1.0",
    )

    registry.set_active(
        "ticket_classifier",
        "2.0",
    )

    assert (
        registry.get_status(
            "ticket_classifier",
            "1.0",
        )
        == PromptStatus.DEPRECATED
    )

    assert (
        registry.get_status(
            "ticket_classifier",
            "2.0",
        )
        == PromptStatus.ACTIVE
    )


def test_deprecated_version_can_be_reactivated_for_rollback() -> None:
    registry = PromptRegistry()

    registry.register(
        make_prompt("1.0"),
    )
    registry.register(
        make_prompt("2.0"),
    )

    registry.set_active(
        "ticket_classifier",
        "1.0",
    )

    registry.set_active(
        "ticket_classifier",
        "2.0",
    )

    registry.set_active(
        "ticket_classifier",
        "1.0",
    )

    assert registry.get_active("ticket_classifier").version == "1.0"

    assert (
        registry.get_status(
            "ticket_classifier",
            "1.0",
        )
        == PromptStatus.ACTIVE
    )

    assert (
        registry.get_status(
            "ticket_classifier",
            "2.0",
        )
        == PromptStatus.DEPRECATED
    )


def test_non_active_prompt_can_be_deprecated() -> None:
    registry = PromptRegistry()

    registry.register(
        make_prompt("1.0"),
    )

    registry.deprecate(
        "ticket_classifier",
        "1.0",
    )

    assert (
        registry.get_status(
            "ticket_classifier",
            "1.0",
        )
        == PromptStatus.DEPRECATED
    )


def test_active_prompt_cannot_be_deprecated_directly() -> None:
    registry = PromptRegistry()

    registry.register(
        make_prompt("1.0"),
    )

    registry.set_active(
        "ticket_classifier",
        "1.0",
    )

    with pytest.raises(
        LLMPromptActiveDeprecationError,
        match="Cannot deprecate",
    ):
        registry.deprecate(
            "ticket_classifier",
            "1.0",
        )
