import pytest

from agent_platform.llm.errors import LLMPromptVariableError
from agent_platform.llm.prompt import PromptTemplate


def test_prompt_template_renders_variables() -> None:
    prompt = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Classify this ticket: {ticket_text}",
    )

    result = prompt.render(
        ticket_text="My invoice was charged twice.",
    )

    assert result == ("Classify this ticket: My invoice was charged twice.")


def test_prompt_template_discovers_required_variables() -> None:
    prompt = PromptTemplate(
        name="customer_summary",
        version="1.0",
        template=("Summarize customer {customer_name} with account {account_id}."),
    )

    assert prompt.required_variables == frozenset(
        {
            "customer_name",
            "account_id",
        }
    )


def test_prompt_template_rejects_missing_variable() -> None:
    prompt = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Classify this ticket: {ticket_text}",
    )

    with pytest.raises(
        LLMPromptVariableError,
        match="ticket_text",
    ):
        prompt.render()


def test_prompt_template_allows_repeated_variable() -> None:
    prompt = PromptTemplate(
        name="repeat_example",
        version="1.0",
        template="{name} has requested a summary for {name}.",
    )

    assert prompt.required_variables == frozenset({"name"})

    result = prompt.render(name="Rachit")

    assert result == ("Rachit has requested a summary for Rachit.")
