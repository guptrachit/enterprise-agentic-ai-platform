import pytest

from agent_platform.llm.api_guardrails import (
    PromptTooLargeError,
    validate_prompt_length,
)


def test_prompt_within_limit_is_allowed() -> None:
    validate_prompt_length(
        "hello",
        max_chars=5,
    )


def test_prompt_at_limit_is_allowed() -> None:
    validate_prompt_length(
        "hello",
        max_chars=5,
    )


def test_prompt_above_limit_is_rejected() -> None:
    with pytest.raises(
        PromptTooLargeError,
        match="exceeds maximum",
    ) as captured:
        validate_prompt_length(
            "hello!",
            max_chars=5,
        )

    assert captured.value.violation.actual_chars == 6

    assert captured.value.violation.max_chars == 5
