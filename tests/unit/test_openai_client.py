import pytest

from agent_platform.config import Settings
from agent_platform.llm.errors import (
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMTransientError,
)
from agent_platform.llm.openai_client import OpenAIClient
from agent_platform.llm.retry import RetryPolicy


class FakeUsage:
    input_tokens = 10
    output_tokens = 5


class FakeResponse:
    id = "resp_test_123"
    output_text = "hello"
    usage = FakeUsage()


@pytest.mark.asyncio
async def test_openai_client_success() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    calls = 0

    async def fake_create(*, model: str, input: str) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    client.client.responses.create = fake_create

    response = await client.generate("hello")

    assert response.text == "hello"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.metadata.provider == "openai"
    assert response.metadata.model == "gpt-5-mini"
    assert response.metadata.request_id == "resp_test_123"
    assert calls == 1


@pytest.mark.asyncio
async def test_openai_client_retries_transient_error() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    client.retry_policy = RetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0,
        jitter=False,
    )

    calls = 0

    async def fake_create(*, model: str, input: str) -> FakeResponse:
        nonlocal calls
        calls += 1

        if calls < 3:
            raise RuntimeError("temporary failure")

        return FakeResponse()

    client.client.responses.create = fake_create

    response = await client.generate("hello")

    assert response.text == "hello"
    assert calls == 3


@pytest.mark.asyncio
async def test_openai_client_does_not_retry_invalid_request() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    calls = 0

    async def fake_create(*, model: str, input: str) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise LLMInvalidRequestError()

    client.client.responses.create = fake_create

    with pytest.raises(LLMInvalidRequestError):
        await client.generate("hello")

    assert calls == 1


@pytest.mark.asyncio
async def test_openai_client_retries_rate_limit() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    client.retry_policy = RetryPolicy(
        max_attempts=2,
        initial_backoff_seconds=0,
        jitter=False,
    )

    calls = 0

    async def fake_create(*, model: str, input: str) -> FakeResponse:
        nonlocal calls
        calls += 1

        if calls == 1:
            raise LLMRateLimitError(
                retry_after_seconds=0,
            )

        return FakeResponse()

    client.client.responses.create = fake_create

    response = await client.generate("hello")

    assert response.text == "hello"
    assert calls == 2


@pytest.mark.asyncio
async def test_openai_client_retries_transient_platform_error() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    client.retry_policy = RetryPolicy(
        max_attempts=2,
        initial_backoff_seconds=0,
        jitter=False,
    )

    calls = 0

    async def fake_create(*, model: str, input: str) -> FakeResponse:
        nonlocal calls
        calls += 1

        if calls == 1:
            raise LLMTransientError()

        return FakeResponse()

    client.client.responses.create = fake_create

    response = await client.generate("hello")

    assert response.text == "hello"
    assert calls == 2


def test_openai_client_uses_configured_retry_policy() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
            llm_max_retries=4,
            llm_initial_backoff_seconds=1.5,
            llm_max_backoff_seconds=7.0,
            llm_timeout_seconds=12.0,
        )
    )

    assert client.retry_policy.max_attempts == 5
    assert client.retry_policy.initial_backoff_seconds == 1.5
    assert client.retry_policy.max_backoff_seconds == 7.0
    assert client.retry_policy.attempt_timeout_seconds == 12.0
