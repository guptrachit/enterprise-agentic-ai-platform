from dataclasses import dataclass


@dataclass(frozen=True)
class ExportMetricsSnapshot:
    """Immutable snapshot of telemetry export metrics."""

    total_exports: int
    successful_exports: int
    failed_exports: int

    @property
    def success_rate(self) -> float:
        """Return successful export fraction."""

        if self.total_exports == 0:
            return 0.0

        return self.successful_exports / self.total_exports

    @property
    def failure_rate(self) -> float:
        """Return failed export fraction."""

        if self.total_exports == 0:
            return 0.0

        return self.failed_exports / self.total_exports

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable export metrics."""

        return {
            "total_exports": self.total_exports,
            "successful_exports": (self.successful_exports),
            "failed_exports": self.failed_exports,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
        }


class ExportMetrics:
    """In-memory metrics for telemetry exporter reliability."""

    def __init__(self) -> None:
        self._total_exports = 0
        self._successful_exports = 0
        self._failed_exports = 0

    def record_success(self) -> None:
        """Record one successful export."""

        self._total_exports += 1
        self._successful_exports += 1

    def record_failure(self) -> None:
        """Record one failed export."""

        self._total_exports += 1
        self._failed_exports += 1

    def snapshot(self) -> ExportMetricsSnapshot:
        """Return a point-in-time metrics snapshot."""

        return ExportMetricsSnapshot(
            total_exports=self._total_exports,
            successful_exports=(self._successful_exports),
            failed_exports=self._failed_exports,
        )
