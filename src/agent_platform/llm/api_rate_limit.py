import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass


class LLMRateLimitExceededError(RuntimeError):
    """Raised when a caller exceeds the configured request rate."""


@dataclass
class _RateLimitWindow:
    started_at: float
    request_count: int


@dataclass(frozen=True)
class RateLimitSnapshot:
    """Point-in-time rate limiter state."""

    max_requests: int
    window_seconds: float
    tracked_callers: int
    rejected_requests: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable limiter state."""

        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "tracked_callers": self.tracked_callers,
            "rejected_requests": self.rejected_requests,
        }


class LLMAPIRateLimiter:
    """Simple in-memory fixed-window rate limiter."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be greater than 0")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock

        self._windows: dict[str, _RateLimitWindow] = {}
        self._rejected_requests = 0
        self._lock = asyncio.Lock()

    async def check(
        self,
        caller_id: str,
    ) -> None:
        """Admit one request or raise when the rate is exceeded."""

        now = self.clock()

        async with self._lock:
            window = self._windows.get(caller_id)

            if window is None or now - window.started_at >= self.window_seconds:
                self._windows[caller_id] = _RateLimitWindow(
                    started_at=now,
                    request_count=1,
                )
                return

            if window.request_count >= self.max_requests:
                self._rejected_requests += 1

                raise LLMRateLimitExceededError("Governed LLM API rate limit exceeded.")

            window.request_count += 1

    def snapshot(
        self,
    ) -> RateLimitSnapshot:
        """Return current rate limiter metrics."""

        return RateLimitSnapshot(
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
            tracked_callers=len(self._windows),
            rejected_requests=self._rejected_requests,
        )
