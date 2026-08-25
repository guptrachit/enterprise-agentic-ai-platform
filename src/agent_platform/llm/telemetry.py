import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

logger = logging.getLogger("agent_platform.llm")


@dataclass(frozen=True)
class LLMExecutionEvent:
    """Normalized telemetry event for an LLM provider execution."""

    timestamp: str
    provider: str
    model: str
    request_id: str | None
    correlation_id: str
    success: bool
    latency_ms: float
    retry_count: int | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    prompt_name: str | None = None
    prompt_version: str | None = None
    workload: str | None = None
    logical_model: str | None = None
    error_type: str | None = None
    policy_identifier: str | None = None
    fallback_used: bool = False
    fallback_from: str | None = None
    fallback_reason: str | None = None
    allowed_providers: tuple[str, ...] | None = None
    max_cost_tier: str | None = None
    max_latency_tier: str | None = None
    prefer_lower_cost: bool = False
    prefer_lower_latency: bool = False
    preferred_providers: tuple[str, ...] | None = None
    preferred_cost_tier: str | None = None
    preferred_latency_tier: str | None = None


@dataclass(frozen=True)
class RoutingDecisionEvent:
    """Structured telemetry describing an LLM routing decision."""

    timestamp: str
    workload: str
    selected_model: str
    ranked_candidates: tuple[str, ...]
    rejected_models: tuple[str, ...]
    routing_reason_codes: tuple[str, ...]
    routing_reasons: tuple[str, ...]
    executed_model: str | None
    fallback_used: bool
    success: bool
    correlation_id: str | None = None
    error_type: str | None = None
    policy_identifier: str | None = None


def create_execution_event(
    *,
    provider: str,
    model: str,
    request_id: str | None,
    correlation_id: str,
    success: bool,
    latency_ms: float,
    retry_count: int | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    workload: str | None = None,
    logical_model: str | None = None,
    error_type: str | None = None,
    fallback_used: bool = False,
    fallback_from: str | None = None,
    fallback_reason: str | None = None,
    allowed_providers: tuple[str, ...] | None = None,
    max_cost_tier: str | None = None,
    max_latency_tier: str | None = None,
    prefer_lower_cost: bool = False,
    prefer_lower_latency: bool = False,
    preferred_providers: tuple[str, ...] | None = None,
    preferred_cost_tier: str | None = None,
    preferred_latency_tier: str | None = None,
) -> LLMExecutionEvent:
    """Create normalized provider-execution telemetry."""

    return LLMExecutionEvent(
        timestamp=datetime.now(UTC).isoformat(),
        provider=provider,
        model=model,
        request_id=request_id,
        correlation_id=correlation_id,
        success=success,
        latency_ms=latency_ms,
        retry_count=retry_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        workload=workload,
        logical_model=logical_model,
        error_type=error_type,
        fallback_used=fallback_used,
        fallback_from=fallback_from,
        fallback_reason=fallback_reason,
        allowed_providers=allowed_providers,
        max_cost_tier=max_cost_tier,
        max_latency_tier=max_latency_tier,
        prefer_lower_cost=prefer_lower_cost,
        prefer_lower_latency=prefer_lower_latency,
        preferred_providers=preferred_providers,
        preferred_cost_tier=preferred_cost_tier,
        preferred_latency_tier=preferred_latency_tier,
    )


def create_routing_decision_event(
    *,
    workload: str,
    selected_model: str,
    ranked_candidates: tuple[str, ...],
    rejected_models: tuple[str, ...],
    routing_reason_codes: tuple[str, ...],
    routing_reasons: tuple[str, ...],
    executed_model: str | None,
    fallback_used: bool,
    success: bool,
    correlation_id: str | None = None,
    error_type: str | None = None,
    policy_identifier: str | None = None,
) -> RoutingDecisionEvent:
    """Create structured model-routing telemetry."""

    return RoutingDecisionEvent(
        timestamp=datetime.now(UTC).isoformat(),
        workload=workload,
        selected_model=selected_model,
        ranked_candidates=ranked_candidates,
        rejected_models=rejected_models,
        routing_reason_codes=routing_reason_codes,
        routing_reasons=routing_reasons,
        executed_model=executed_model,
        fallback_used=fallback_used,
        success=success,
        correlation_id=correlation_id,
        error_type=error_type,
        policy_identifier=policy_identifier,
    )


def log_execution_event(event: LLMExecutionEvent) -> None:
    """Write an LLM execution event as structured JSON."""

    logger.info(
        "llm_execution %s",
        json.dumps(
            asdict(event),
            sort_keys=True,
        ),
    )


def log_routing_decision_event(
    event: RoutingDecisionEvent,
) -> None:
    """Write an LLM routing decision as structured JSON."""

    logger.info(
        "llm_routing_decision %s",
        json.dumps(
            asdict(event),
            sort_keys=True,
        ),
    )
