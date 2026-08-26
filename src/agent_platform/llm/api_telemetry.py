import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

logger = logging.getLogger("agent_platform.llm")


@dataclass(frozen=True)
class LLMAPIRequestEvent:
    """Structured telemetry for one governed LLM API request."""

    timestamp: str
    correlation_id: str
    status_code: int
    latency_ms: float
    success: bool
    policy_identifier: str | None
    model: str | None
    provider: str | None
    failure_code: str | None


def create_llm_api_request_event(
    *,
    correlation_id: str,
    status_code: int,
    latency_ms: float,
    success: bool,
    policy_identifier: str | None,
    model: str | None,
    provider: str | None,
    failure_code: str | None = None,
) -> LLMAPIRequestEvent:
    """Create structured API telemetry."""

    return LLMAPIRequestEvent(
        timestamp=datetime.now(UTC).isoformat(),
        correlation_id=correlation_id,
        status_code=status_code,
        latency_ms=latency_ms,
        success=success,
        policy_identifier=policy_identifier,
        model=model,
        provider=provider,
        failure_code=failure_code,
    )


def log_llm_api_request_event(
    event: LLMAPIRequestEvent,
) -> None:
    """Log structured governed LLM API telemetry."""

    logger.info(
        "llm_api_request %s",
        json.dumps(
            asdict(event),
            sort_keys=True,
        ),
    )
