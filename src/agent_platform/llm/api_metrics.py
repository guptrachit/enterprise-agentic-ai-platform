from dataclasses import dataclass


@dataclass(frozen=True)
class LLMAPIMetricsSnapshot:
    """Immutable snapshot of governed LLM API metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    status_counts: dict[int, int]
    failure_counts: dict[str, int]
    total_latency_ms: float

    @property
    def average_latency_ms(self) -> float:
        """Return average API latency."""

        if self.total_requests == 0:
            return 0.0

        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        """Return successful request fraction."""

        if self.total_requests == 0:
            return 0.0

        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        """Return failed request fraction."""

        if self.total_requests == 0:
            return 0.0

        return self.failed_requests / self.total_requests

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable API metrics."""

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "status_counts": dict(self.status_counts),
            "failure_counts": dict(self.failure_counts),
            "total_latency_ms": self.total_latency_ms,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
        }


class LLMAPIMetrics:
    """In-memory metrics for governed LLM API requests."""

    def __init__(self) -> None:
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._status_counts: dict[int, int] = {}
        self._failure_counts: dict[str, int] = {}
        self._total_latency_ms = 0.0

    def record_request(
        self,
        *,
        status_code: int,
        latency_ms: float,
        success: bool,
        failure_code: str | None = None,
    ) -> None:
        """Record one governed LLM API request."""

        self._total_requests += 1
        self._total_latency_ms += latency_ms

        self._status_counts[status_code] = (
            self._status_counts.get(
                status_code,
                0,
            )
            + 1
        )

        if success:
            self._successful_requests += 1
            return

        self._failed_requests += 1

        if failure_code is not None:
            self._failure_counts[failure_code] = (
                self._failure_counts.get(
                    failure_code,
                    0,
                )
                + 1
            )

    def snapshot(self) -> LLMAPIMetricsSnapshot:
        """Return an immutable point-in-time metrics snapshot."""

        return LLMAPIMetricsSnapshot(
            total_requests=self._total_requests,
            successful_requests=self._successful_requests,
            failed_requests=self._failed_requests,
            status_counts=dict(self._status_counts),
            failure_counts=dict(self._failure_counts),
            total_latency_ms=self._total_latency_ms,
        )
