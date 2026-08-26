from agent_platform.llm.api_guardrails import (
    InFlightRequestGuard,
)
from agent_platform.llm.api_health_status import (
    assess_llm_api_health,
)
from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)
from agent_platform.llm.api_rate_limit import (
    LLMAPIRateLimiter,
)
from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)
from agent_platform.observability.export_metrics import (
    ExportMetrics,
)
from agent_platform.observability.exporter import (
    OperationalTelemetryExporter,
)
from agent_platform.observability.logging_exporter import (
    LoggingOperationalTelemetryExporter,
)
from agent_platform.observability.operational_snapshot import (
    OperationalObservabilitySnapshot,
)
from agent_platform.observability.operational_telemetry import (
    create_operational_telemetry_envelope,
)
from agent_platform.observability.resilient_exporter import (
    ResilientOperationalTelemetryExporter,
)
from agent_platform.security.security_metrics import (
    SecurityMetrics,
)


class OperationalObservabilityService:
    """Build and export unified operational observability snapshots."""

    def __init__(
        self,
        *,
        runtime: RefreshAwareLLMExecutionService,
        api_metrics: LLMAPIMetrics,
        concurrency_guard: InFlightRequestGuard,
        rate_limiter: LLMAPIRateLimiter,
        security_metrics: SecurityMetrics,
        exporter: OperationalTelemetryExporter | None = None,
        export_metrics: ExportMetrics | None = None,
    ) -> None:
        self._runtime = runtime
        self._api_metrics = api_metrics
        self._concurrency_guard = concurrency_guard
        self._rate_limiter = rate_limiter
        self._security_metrics = security_metrics

        self._export_metrics = export_metrics or ExportMetrics()

        configured_exporter = exporter or LoggingOperationalTelemetryExporter()

        self._exporter = ResilientOperationalTelemetryExporter(
            configured_exporter,
            metrics=self._export_metrics,
        )

    @property
    def export_metrics(self) -> ExportMetrics:
        """Return operational exporter reliability metrics."""

        return self._export_metrics

    def snapshot(
        self,
    ) -> OperationalObservabilitySnapshot:
        """Create one consistent point-in-time operational snapshot."""

        runtime = self._runtime.health_snapshot().to_dict()

        api = self._api_metrics.snapshot().to_dict()

        concurrency = self._concurrency_guard.snapshot().to_dict()

        rate_limit = self._rate_limiter.snapshot().to_dict()

        security = self._security_metrics.snapshot().to_dict()

        export = self._export_metrics.snapshot().to_dict()

        assessment = assess_llm_api_health(
            runtime_resolved=bool(
                runtime.get(
                    "resolved",
                    False,
                )
            ),
            utilization_rate=float(concurrency["utilization_rate"]),
            capacity_rejections=int(concurrency["capacity_rejections"]),
            failure_counts=dict(api["failure_counts"]),
            export_failed_count=int(export["failed_exports"]),
        )

        health = {
            "status": assessment.status.value,
            "reasons": list(assessment.reasons),
        }

        return OperationalObservabilitySnapshot(
            health=health,
            runtime=runtime,
            api=api,
            concurrency=concurrency,
            rate_limit=rate_limit,
            security=security,
            export=export,
        )

    def export_snapshot(
        self,
    ) -> OperationalObservabilitySnapshot:
        """Create and safely export one operational snapshot."""

        snapshot = self.snapshot()

        envelope = create_operational_telemetry_envelope(snapshot)

        self._exporter.export(envelope)

        return snapshot
