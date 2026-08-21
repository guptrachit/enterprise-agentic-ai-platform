import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent_platform.llm.errors import (
    LLMError,
    LLMRetryBudgetExceededError,
    LLMTimeoutError,
)
from agent_platform.llm.retry import RetryPolicy


@dataclass(frozen=True)
class RetryExecutionResult[T]:
    """Result of an operation executed with retry protection."""

    result: T
    attempts: int

    @property
    def retry_count(self) -> int:
        """Return the number of retries performed."""
        return self.attempts - 1


async def execute_with_retry[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
) -> RetryExecutionResult[T]:
    """Execute an async operation within a retry time budget."""

    start_time = time.monotonic()

    for attempt in range(policy.max_attempts):
        try:
            result = await asyncio.wait_for(
                operation(),
                timeout=policy.attempt_timeout_seconds,
            )

            return RetryExecutionResult(
                result=result,
                attempts=attempt + 1,
            )

        except asyncio.CancelledError:
            raise

        except TimeoutError as error:
            llm_error = LLMTimeoutError()
            llm_error.__cause__ = error

        except LLMError as error:
            llm_error = error

        if not llm_error.retryable:
            raise llm_error

        if attempt == policy.max_attempts - 1:
            raise llm_error

        elapsed = time.monotonic() - start_time
        remaining_budget = policy.max_retry_budget_seconds - elapsed

        if remaining_budget <= 0:
            raise LLMRetryBudgetExceededError() from llm_error

        delay = policy.retry_delay(
            retry_number=attempt,
            retry_after_seconds=llm_error.retry_after_seconds,
        )

        if delay > remaining_budget:
            raise LLMRetryBudgetExceededError(
                "LLM retry budget would be exceeded by the next retry delay."
            ) from llm_error

        await asyncio.sleep(delay)

    raise RuntimeError("Retry execution reached an unexpected state.")
