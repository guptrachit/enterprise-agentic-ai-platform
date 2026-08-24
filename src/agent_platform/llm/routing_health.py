from dataclasses import dataclass
from enum import StrEnum

from agent_platform.llm.routing_metrics import RoutingMetricsSnapshot


class RoutingHealthStatus(StrEnum):
    """Operational health status for LLM routing."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class RoutingHealthSnapshot:
    """Point-in-time operational health assessment."""

    status: RoutingHealthStatus
    total_requests: int
    failure_rate: float
    fallback_rate: float
    metrics_export_failures: int
    reasons: tuple[str, ...]


def evaluate_routing_health(
    snapshot: RoutingMetricsSnapshot,
) -> RoutingHealthSnapshot:
    """Evaluate routing health from aggregate routing metrics."""

    reasons: list[str] = []

    if snapshot.failure_rate >= 0.50:
        reasons.append("Routing failure rate is at or above 50%.")

        return RoutingHealthSnapshot(
            status=RoutingHealthStatus.UNHEALTHY,
            total_requests=snapshot.total_requests,
            failure_rate=snapshot.failure_rate,
            fallback_rate=snapshot.fallback_rate,
            metrics_export_failures=snapshot.metrics_export_failures,
            reasons=tuple(reasons),
        )

    if snapshot.failure_rate > 0:
        reasons.append("One or more routing requests failed.")

    if snapshot.fallback_rate >= 0.25:
        reasons.append("Routing fallback rate is at or above 25%.")

    if snapshot.metrics_export_failures > 0:
        reasons.append("One or more routing metrics exports failed.")

    status = RoutingHealthStatus.DEGRADED if reasons else RoutingHealthStatus.HEALTHY

    return RoutingHealthSnapshot(
        status=status,
        total_requests=snapshot.total_requests,
        failure_rate=snapshot.failure_rate,
        fallback_rate=snapshot.fallback_rate,
        metrics_export_failures=snapshot.metrics_export_failures,
        reasons=tuple(reasons),
    )
