import asyncio

import pytest

from agent_platform.tools import (
    ToolNonRetryableError,
    ToolRetryableError,
    ToolRetryExecutor,
    ToolRetryExhaustedError,
    ToolRetryPolicy,
)


def test_retry_policy_rejects_invalid_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ToolRetryPolicy(
            max_attempts=0,
        )


def test_retry_policy_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ToolRetryPolicy(
            timeout_seconds=0,
        )


def test_retry_policy_rejects_negative_retry_delay() -> None:
    with pytest.raises(ValueError, match="retry_delay_seconds"):
        ToolRetryPolicy(
            retry_delay_seconds=-1,
        )


@pytest.mark.asyncio
async def test_retry_executor_returns_successful_result() -> None:
    executor = ToolRetryExecutor(
        policy=ToolRetryPolicy(
            max_attempts=3,
            timeout_seconds=1,
        )
    )

    async def operation() -> str:
        return "success"

    result = await executor.execute(operation)

    assert result == "success"


@pytest.mark.asyncio
async def test_retry_executor_retries_retryable_failure() -> None:
    executor = ToolRetryExecutor(
        policy=ToolRetryPolicy(
            max_attempts=3,
            timeout_seconds=1,
        )
    )

    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise ToolRetryableError("temporary failure")

        return "success"

    result = await executor.execute(operation)

    assert result == "success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_executor_does_not_retry_non_retryable_failure() -> None:
    executor = ToolRetryExecutor(
        policy=ToolRetryPolicy(
            max_attempts=3,
            timeout_seconds=1,
        )
    )

    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ToolNonRetryableError("permanent failure")

    with pytest.raises(
        ToolNonRetryableError,
        match="permanent failure",
    ):
        await executor.execute(operation)

    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_executor_exhausts_retryable_failures() -> None:
    executor = ToolRetryExecutor(
        policy=ToolRetryPolicy(
            max_attempts=2,
            timeout_seconds=1,
        )
    )

    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ToolRetryableError("temporary failure")

    with pytest.raises(ToolRetryExhaustedError):
        await executor.execute(operation)

    assert attempts == 2


@pytest.mark.asyncio
async def test_retry_executor_times_out_and_exhausts() -> None:
    executor = ToolRetryExecutor(
        policy=ToolRetryPolicy(
            max_attempts=2,
            timeout_seconds=0.01,
        )
    )

    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.1)
        return "never-reached"

    with pytest.raises(ToolRetryExhaustedError):
        await executor.execute(operation)

    assert attempts == 2
