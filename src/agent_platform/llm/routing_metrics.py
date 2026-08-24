from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class RoutingMetricsSnapshot:
    """Immutable point-in-time snapshot of routing metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    fallback_requests: int
    metrics_export_failures: int

    success_rate: float
    failure_rate: float
    fallback_rate: float

    model_selection_counts: Mapping[str, int]
    executed_model_counts: Mapping[str, int]
    rejection_reason_counts: Mapping[str, int]
    workload_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "fallback_requests": self.fallback_requests,
            "metrics_export_failures": self.metrics_export_failures,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "fallback_rate": self.fallback_rate,
            "model_selection_counts": dict(self.model_selection_counts),
            "executed_model_counts": dict(self.executed_model_counts),
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "workload_counts": dict(self.workload_counts),
        }


@dataclass
class RoutingMetrics:
    """In-memory aggregate metrics for LLM routing outcomes."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    fallback_requests: int = 0
    metrics_export_failures: int = 0

    model_selection_counts: dict[str, int] = field(default_factory=dict)
    executed_model_counts: dict[str, int] = field(default_factory=dict)
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    workload_counts: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Return the fraction of requests that succeeded."""

        if self.total_requests == 0:
            return 0.0

        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        """Return the fraction of requests that failed."""

        if self.total_requests == 0:
            return 0.0

        return self.failed_requests / self.total_requests

    @property
    def fallback_rate(self) -> float:
        """Return the fraction of requests that used fallback."""

        if self.total_requests == 0:
            return 0.0

        return self.fallback_requests / self.total_requests

    def record_request(
        self,
        *,
        workload: str,
        selected_model: str,
        executed_model: str | None,
        routing_reason_codes: tuple[str, ...],
        success: bool,
        fallback_used: bool,
    ) -> None:
        """Record one completed routing outcome."""

        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if fallback_used:
            self.fallback_requests += 1

        self.model_selection_counts[selected_model] = (
            self.model_selection_counts.get(
                selected_model,
                0,
            )
            + 1
        )

        if executed_model is not None:
            self.executed_model_counts[executed_model] = (
                self.executed_model_counts.get(
                    executed_model,
                    0,
                )
                + 1
            )

        for reason_code in routing_reason_codes:
            self.rejection_reason_counts[reason_code] = (
                self.rejection_reason_counts.get(
                    reason_code,
                    0,
                )
                + 1
            )

        self.workload_counts[workload] = (
            self.workload_counts.get(
                workload,
                0,
            )
            + 1
        )

    def record_export_failure(self) -> None:
        """Record one routing metrics exporter failure."""

        self.metrics_export_failures += 1

    def snapshot(self) -> RoutingMetricsSnapshot:
        """Return an immutable point-in-time metrics snapshot."""

        return RoutingMetricsSnapshot(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            fallback_requests=self.fallback_requests,
            metrics_export_failures=self.metrics_export_failures,
            success_rate=self.success_rate,
            failure_rate=self.failure_rate,
            fallback_rate=self.fallback_rate,
            model_selection_counts=MappingProxyType(self.model_selection_counts.copy()),
            executed_model_counts=MappingProxyType(self.executed_model_counts.copy()),
            rejection_reason_counts=MappingProxyType(
                self.rejection_reason_counts.copy()
            ),
            workload_counts=MappingProxyType(self.workload_counts.copy()),
        )
