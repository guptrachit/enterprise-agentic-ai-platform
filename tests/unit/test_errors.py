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
