import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLengthViolation:
    """Describes an oversized LLM prompt."""

    actual_chars: int
    max_chars: int


class PromptTooLargeError(ValueError):
    """Raised when an API prompt exceeds the configured maximum."""

    def __init__(
        self,
        *,
        actual_chars: int,
        max_chars: int,
    ) -> None:
        self.violation = PromptLengthViolation(
            actual_chars=actual_chars,
            max_chars=max_chars,
        )

        super().__init__(
            f"Prompt length {actual_chars} exceeds maximum {max_chars} characters."
        )


class LLMCapacityExceededError(RuntimeError):
    """Raised when the governed LLM API is at in-flight capacity."""


@dataclass(frozen=True)
class InFlightRequestGuardSnapshot:
    """Point-in-time concurrency guard metrics."""

    max_in_flight: int
    active_requests: int
    available_capacity: int
    capacity_rejections: int

    @property
    def utilization_rate(self) -> float:
        """Return current fraction of configured capacity in use."""

        if self.max_in_flight == 0:
            return 0.0

        return self.active_requests / self.max_in_flight

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable concurrency guard metrics."""

        return {
            "max_in_flight": self.max_in_flight,
            "active_requests": self.active_requests,
            "available_capacity": self.available_capacity,
            "capacity_rejections": self.capacity_rejections,
            "utilization_rate": self.utilization_rate,
        }


class InFlightRequestGuard:
    """Bound concurrent governed LLM API executions."""

    def __init__(
        self,
        *,
        max_in_flight: int,
    ) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be greater than 0")

        self.max_in_flight = max_in_flight
        self._active_requests = 0
        self._capacity_rejections = 0
        self._lock = asyncio.Lock()

    @property
    def active_requests(self) -> int:
        """Return the number of currently admitted requests."""

        return self._active_requests

    @property
    def available_capacity(self) -> int:
        """Return currently available request capacity."""

        return max(
            self.max_in_flight - self._active_requests,
            0,
        )

    @property
    def capacity_rejections(self) -> int:
        """Return number of rejected requests due to capacity."""

        return self._capacity_rejections

    def snapshot(
        self,
    ) -> InFlightRequestGuardSnapshot:
        """Return current concurrency guard metrics."""

        return InFlightRequestGuardSnapshot(
            max_in_flight=self.max_in_flight,
            active_requests=self._active_requests,
            available_capacity=self.available_capacity,
            capacity_rejections=self._capacity_rejections,
        )

    @asynccontextmanager
    async def slot(
        self,
    ) -> AsyncIterator[None]:
        """Acquire one in-flight slot or reject immediately."""

        async with self._lock:
            if self._active_requests >= self.max_in_flight:
                self._capacity_rejections += 1

                raise LLMCapacityExceededError(
                    "Governed LLM API capacity is exhausted."
                )

            self._active_requests += 1

        try:
            yield
        finally:
            async with self._lock:
                self._active_requests -= 1


def validate_prompt_length(
    prompt: str,
    *,
    max_chars: int,
) -> None:
    """Reject prompts larger than the configured maximum."""

    actual_chars = len(prompt)

    if actual_chars > max_chars:
        raise PromptTooLargeError(
            actual_chars=actual_chars,
            max_chars=max_chars,
        )
