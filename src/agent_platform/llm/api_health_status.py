from dataclasses import dataclass
from enum import StrEnum


class LLMAPIHealthStatus(StrEnum):
    """Operational health state for the governed LLM API."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    SATURATED = "saturated"


@dataclass(frozen=True)
class LLMAPIHealthAssessment:
    """Health classification with operational reason codes."""

    status: LLMAPIHealthStatus
    reasons: tuple[str, ...]


def assess_llm_api_health(
    *,
    runtime_resolved: bool,
    utilization_rate: float,
    capacity_rejections: int,
    failure_counts: dict[str, int],
) -> LLMAPIHealthAssessment:
    """Assess governed LLM API health and reason codes."""

    reasons: list[str] = []

    if utilization_rate >= 1.0:
        reasons.append("llm_capacity_saturated")

    if capacity_rejections > 0:
        reasons.append("llm_capacity_exceeded")

    if reasons:
        return LLMAPIHealthAssessment(
            status=LLMAPIHealthStatus.SATURATED,
            reasons=tuple(reasons),
        )

    if not runtime_resolved:
        reasons.append("routing_policy_unresolved")

    degraded_failure_codes = (
        "llm_request_timeout",
        "llm_temporarily_unavailable",
        "llm_model_unavailable",
        "routing_policy_unavailable",
        "llm_rate_limit_exceeded",
    )

    for code in degraded_failure_codes:
        if (
            failure_counts.get(
                code,
                0,
            )
            > 0
        ):
            reasons.append(code)

    if reasons:
        return LLMAPIHealthAssessment(
            status=LLMAPIHealthStatus.DEGRADED,
            reasons=tuple(reasons),
        )

    return LLMAPIHealthAssessment(
        status=LLMAPIHealthStatus.HEALTHY,
        reasons=(),
    )


def classify_llm_api_health(
    *,
    runtime_resolved: bool,
    utilization_rate: float,
    capacity_rejections: int,
    failure_counts: dict[str, int],
) -> LLMAPIHealthStatus:
    """Return only the governed LLM API health status."""

    return assess_llm_api_health(
        runtime_resolved=runtime_resolved,
        utilization_rate=utilization_rate,
        capacity_rejections=capacity_rejections,
        failure_counts=failure_counts,
    ).status
