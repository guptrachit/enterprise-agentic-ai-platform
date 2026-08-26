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
from agent_platform.security.security_metrics import (
    SecurityMetrics,
)


def create_llm_api_health_payload(
    *,
    runtime: RefreshAwareLLMExecutionService,
    api_metrics: LLMAPIMetrics,
    concurrency_guard: InFlightRequestGuard,
    rate_limiter: LLMAPIRateLimiter,
    security_metrics: SecurityMetrics,
) -> dict[str, object]:
    """Create safe operational health for the governed LLM API."""

    runtime_snapshot = runtime.health_snapshot().to_dict()

    api_metrics_snapshot = api_metrics.snapshot().to_dict()

    concurrency_snapshot = concurrency_guard.snapshot().to_dict()

    rate_limit_snapshot = rate_limiter.snapshot().to_dict()

    security_snapshot = security_metrics.snapshot().to_dict()

    assessment = assess_llm_api_health(
        runtime_resolved=bool(
            runtime_snapshot.get(
                "resolved",
                False,
            )
        ),
        utilization_rate=float(concurrency_snapshot["utilization_rate"]),
        capacity_rejections=int(concurrency_snapshot["capacity_rejections"]),
        failure_counts=dict(api_metrics_snapshot["failure_counts"]),
    )

    return {
        "status": assessment.status.value,
        "reasons": list(assessment.reasons),
        "runtime": runtime_snapshot,
        "api_metrics": api_metrics_snapshot,
        "concurrency": concurrency_snapshot,
        "rate_limit": rate_limit_snapshot,
        "security": security_snapshot,
    }
