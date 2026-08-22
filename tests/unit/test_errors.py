from agent_platform.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransientError,
)


def test_llm_errors_inherit_from_base_error() -> None:
    errors = [
        LLMTimeoutError,
        LLMRateLimitError,
        LLMTransientError,
        LLMConfigurationError,
        LLMInvalidRequestError,
    ]

    for error_type in errors:
        assert issubclass(error_type, LLMError)


def test_structured_validation_error_is_not_retryable() -> None:
    from agent_platform.llm.errors import LLMStructuredValidationError

    error = LLMStructuredValidationError()

    assert error.retryable is False


def test_structured_parse_error_is_not_retryable() -> None:
    from agent_platform.llm.errors import LLMStructuredParseError

    error = LLMStructuredParseError()

    assert error.retryable is False


def test_refusal_error_is_not_retryable() -> None:
    from agent_platform.llm.errors import LLMRefusalError

    error = LLMRefusalError()

    assert error.retryable is False
