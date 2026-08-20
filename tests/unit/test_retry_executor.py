import asyncio

import pytest

from agent_platform.llm.errors import (
    LLMInvalidRequestError,
    LLMRetryBudgetExceededError,
    LLMTimeoutError,
    LLMTransientError,
)
from agent_platform.llm.retry import RetryPolicy
from agent_platform.llm.retry_executor import execute_with_retry


@pytest.mark.asyncio
async def test_successful_operation_is_not_retried() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "success"

    result = await execute_with_retry(
        operation,
        RetryPolicy(jitter=False),
    )

    assert result == "success"
    assert calls == 1


@pytest.mark.asyncio
async def test_transient_error_is_retried() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1

        if calls < 3:
            raise LLMTransientError()

        return "success"

    result = await execute_with_retry(
        operation,
        RetryPolicy(
            initial_backoff_seconds=0,
            jitter=False,
        ),
    )

    assert result == "success"
    assert calls == 3


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise LLMInvalidRequestError()

    with pytest.raises(LLMInvalidRequestError):
        await execute_with_retry(
            operation,
            RetryPolicy(
                initial_backoff_seconds=0,
                jitter=False,
            ),
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_retry_limit_is_respected() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise LLMTransientError()

    with pytest.raises(LLMTransientError):
        await execute_with_retry(
            operation,
            RetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=0,
                jitter=False,
            ),
        )

    assert calls == 3


@pytest.mark.asyncio
async def test_retry_budget_stops_long_retry_delay() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise LLMTransientError(
            retry_after_seconds=5,
        )

    with pytest.raises(LLMRetryBudgetExceededError):
        await execute_with_retry(
            operation,
            RetryPolicy(
                max_attempts=3,
                max_retry_budget_seconds=1,
                jitter=False,
            ),
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_retry_budget_allows_retry_when_delay_fits() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1

        if calls == 1:
            raise LLMTransientError(
                retry_after_seconds=0,
            )

        return "success"

    result = await execute_with_retry(
        operation,
        RetryPolicy(
            max_attempts=3,
            max_retry_budget_seconds=1,
            jitter=False,
        ),
    )

    assert result == "success"
    assert calls == 2


@pytest.mark.asyncio
async def test_timeout_is_retried() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return "success"

    with pytest.raises(LLMTimeoutError):
        await execute_with_retry(
            operation,
            RetryPolicy(
                max_attempts=2,
                attempt_timeout_seconds=0.01,
                initial_backoff_seconds=0,
                jitter=False,
            ),
        )

    assert calls == 2
