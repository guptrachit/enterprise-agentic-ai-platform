import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

logger = logging.getLogger("agent_platform.llm")


@dataclass(frozen=True)
class LLMExecutionEvent:
    """Normalized telemetry event for an LLM execution."""

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
    error_type: str | None = None


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
    error_type: str | None = None,
) -> LLMExecutionEvent:
    """Create a normalized LLM execution telemetry event."""

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
        error_type=error_type,
    )


def log_execution_event(event: LLMExecutionEvent) -> None:
    """Write an LLM execution event as structured JSON."""

    logger.info(
        "llm_execution %s",
        json.dumps(asdict(event), sort_keys=True),
    )
