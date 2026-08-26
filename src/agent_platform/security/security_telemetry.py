import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger("agent_platform.security")


class SecurityEventType(StrEnum):
    """Supported application security telemetry event types."""

    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"


@dataclass(frozen=True)
class SecurityEvent:
    """Safe structured telemetry for one security event."""

    timestamp: str
    correlation_id: str
    event_type: SecurityEventType
    status_code: int
    failure_code: str
    path: str


def create_security_event(
    *,
    correlation_id: str,
    event_type: SecurityEventType,
    status_code: int,
    failure_code: str,
    path: str,
) -> SecurityEvent:
    """Create safe structured security telemetry."""

    return SecurityEvent(
        timestamp=datetime.now(UTC).isoformat(),
        correlation_id=correlation_id,
        event_type=event_type,
        status_code=status_code,
        failure_code=failure_code,
        path=path,
    )


def log_security_event(
    event: SecurityEvent,
) -> None:
    """Log one structured application security event."""

    logger.info(
        "security_event %s",
        json.dumps(
            asdict(event),
            sort_keys=True,
        ),
    )
