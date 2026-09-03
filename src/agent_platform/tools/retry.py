import asyncio
from dataclasses import dataclass


class ToolInvocationError(Exception):
    """Base error raised when the underlying tool invocation fails."""


class ToolRetryableError(ToolInvocationError):
    """Failure that may be retried according to policy."""


class ToolNonRetryableError(ToolInvocationError):
    """Failure that must not be retried."""


class ToolTimeoutError(ToolInvocationError):
    """Raised when tool execution exceeds the configured timeout."""


class ToolRetryExhaustedError(ToolInvocationError):
    """Raised when all configured execution attempts are exhausted."""


@dataclass(frozen=True)
class ToolRetryPolicy:
    """Bounded retry and timeout policy for tool execution."""

    max_attempts: int = 1
    timeout_seconds: float = 30.0
    retry_delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")


class ToolRetryExecutor:
    """Executes tool operations with timeout and bounded retry semantics."""

    def __init__(
        self,
        *,
        policy: ToolRetryPolicy,
    ) -> None:
        self._policy = policy

    async def execute(self, operation):
        last_error: ToolInvocationError | None = None

        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                async with asyncio.timeout(self._policy.timeout_seconds):
                    return await operation()

            except TimeoutError:
                timeout_error = ToolTimeoutError(
                    f"tool execution timed out after "
                    f"{self._policy.timeout_seconds} seconds"
                )

                last_error = timeout_error

            except ToolRetryableError as exc:
                last_error = exc

            except ToolNonRetryableError:
                raise

            if (
                attempt < self._policy.max_attempts
                and self._policy.retry_delay_seconds > 0
            ):
                await asyncio.sleep(self._policy.retry_delay_seconds)

        raise ToolRetryExhaustedError(
            f"tool execution failed after {self._policy.max_attempts} attempt(s)"
        ) from last_error
