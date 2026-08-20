import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration controlling LLM retry behavior."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 5.0
    max_retry_budget_seconds: float = 10.0
    attempt_timeout_seconds: float = 30.0
    jitter: bool = True

    def backoff_seconds(self, retry_number: int) -> float:
        """Calculate exponential backoff with optional jitter."""

        delay = self.initial_backoff_seconds * (2**retry_number)
        delay = min(delay, self.max_backoff_seconds)

        if self.jitter:
            delay = random.uniform(0, delay)

        return delay

    def retry_delay(
        self,
        retry_number: int,
        retry_after_seconds: float | None = None,
    ) -> float:
        """Determine retry delay using provider guidance when available."""

        if retry_after_seconds is not None:
            return max(0.0, retry_after_seconds)

        return self.backoff_seconds(retry_number)
