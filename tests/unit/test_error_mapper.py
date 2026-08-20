import httpx2
import openai

from agent_platform.llm.error_mapper import map_openai_error
from agent_platform.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransientError,
)


def create_response(status_code: int) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    return httpx2.Response(status_code, request=request)


def test_timeout_error_mapping() -> None:
    error = openai.APITimeoutError(
        request=httpx2.Request(
            "POST",
            "https://api.openai.com/v1/responses",
        )
    )

    mapped = map_openai_error(error)

    assert isinstance(mapped, LLMTimeoutError)


def test_rate_limit_error_mapping() -> None:
    error = openai.RateLimitError(
        message="rate limited",
        response=create_response(429),
        body=None,
    )

    mapped = map_openai_error(error)

    assert isinstance(mapped, LLMRateLimitError)


def test_authentication_error_mapping() -> None:
    error = openai.AuthenticationError(
        message="authentication failed",
        response=create_response(401),
        body=None,
    )

    mapped = map_openai_error(error)

    assert isinstance(mapped, LLMConfigurationError)


def test_bad_request_error_mapping() -> None:
    error = openai.BadRequestError(
        message="bad request",
        response=create_response(400),
        body=None,
    )

    mapped = map_openai_error(error)

    assert isinstance(mapped, LLMInvalidRequestError)


def test_internal_server_error_mapping() -> None:
    error = openai.InternalServerError(
        message="server error",
        response=create_response(500),
        body=None,
    )

    mapped = map_openai_error(error)

    assert isinstance(mapped, LLMTransientError)


def test_unknown_error_mapping() -> None:
    mapped = map_openai_error(RuntimeError("unexpected"))

    assert isinstance(mapped, LLMError)


def test_rate_limit_retry_after_is_preserved() -> None:
    request = httpx2.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )
    response = httpx2.Response(
        429,
        headers={"Retry-After": "20"},
        request=request,
    )

    error = openai.RateLimitError(
        message="rate limited",
        response=response,
        body=None,
    )

    mapped = map_openai_error(error)

    assert isinstance(mapped, LLMRateLimitError)
    assert mapped.retryable is True
    assert mapped.retry_after_seconds == 20.0


def test_retry_after_missing() -> None:
    error = openai.RateLimitError(
        message="rate limited",
        response=create_response(429),
        body=None,
    )

    mapped = map_openai_error(error)

    assert isinstance(mapped, LLMRateLimitError)
    assert mapped.retryable is True
    assert mapped.retry_after_seconds is None
