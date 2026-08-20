import openai

from agent_platform.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransientError,
)


def _retry_after_seconds(error: Exception) -> float | None:
    """Extract Retry-After from an OpenAI HTTP response."""

    response = getattr(error, "response", None)

    if response is None:
        return None

    value = response.headers.get("retry-after")

    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def map_openai_error(error: Exception) -> LLMError:
    """Translate OpenAI SDK exceptions into platform-level LLM errors."""

    retry_after = _retry_after_seconds(error)

    if isinstance(error, openai.APITimeoutError):
        return LLMTimeoutError()

    if isinstance(error, openai.RateLimitError):
        return LLMRateLimitError(
            retry_after_seconds=retry_after,
        )

    if isinstance(error, openai.AuthenticationError):
        return LLMConfigurationError("OpenAI authentication failed.")

    if isinstance(error, openai.BadRequestError):
        return LLMInvalidRequestError("OpenAI rejected the request.")

    if isinstance(error, (openai.APIConnectionError, openai.InternalServerError)):
        return LLMTransientError(
            retry_after_seconds=retry_after,
        )

    return LLMTransientError("OpenAI request failed.")
