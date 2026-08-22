from agent_platform.llm.prompt import PromptTemplate
from agent_platform.llm.prompt_environment import PromptEnvironment
from agent_platform.llm.prompt_registry import PromptRegistry


def make_prompt(version: str) -> PromptTemplate:
    return PromptTemplate(
        name="ticket_classifier",
        version=version,
        template="Classify: {ticket_text}",
    )


def test_different_environments_can_use_different_versions() -> None:
    registry = PromptRegistry()

    registry.register(make_prompt("1.0"))
    registry.register(make_prompt("2.0"))
    registry.register(make_prompt("3.0"))

    registry.set_active(
        "ticket_classifier",
        "3.0",
        environment=PromptEnvironment.DEVELOPMENT,
    )

    registry.set_active(
        "ticket_classifier",
        "2.0",
        environment=PromptEnvironment.QA,
    )

    registry.set_active(
        "ticket_classifier",
        "1.0",
        environment=PromptEnvironment.PRODUCTION,
    )

    assert (
        registry.get_active(
            "ticket_classifier",
            environment=PromptEnvironment.DEVELOPMENT,
        ).version
        == "3.0"
    )

    assert (
        registry.get_active(
            "ticket_classifier",
            environment=PromptEnvironment.QA,
        ).version
        == "2.0"
    )

    assert (
        registry.get_active(
            "ticket_classifier",
            environment=PromptEnvironment.PRODUCTION,
        ).version
        == "1.0"
    )


def test_default_environment_is_production() -> None:
    registry = PromptRegistry()

    registry.register(make_prompt("1.0"))

    registry.set_active(
        "ticket_classifier",
        "1.0",
    )

    assert registry.get_active("ticket_classifier").version == "1.0"


def test_promoting_in_qa_does_not_change_production() -> None:
    registry = PromptRegistry()

    registry.register(make_prompt("1.0"))
    registry.register(make_prompt("2.0"))

    registry.set_active(
        "ticket_classifier",
        "1.0",
        environment=PromptEnvironment.PRODUCTION,
    )

    registry.set_active(
        "ticket_classifier",
        "2.0",
        environment=PromptEnvironment.QA,
    )

    assert (
        registry.get_active(
            "ticket_classifier",
            environment=PromptEnvironment.PRODUCTION,
        ).version
        == "1.0"
    )

    assert (
        registry.get_active(
            "ticket_classifier",
            environment=PromptEnvironment.QA,
        ).version
        == "2.0"
    )
