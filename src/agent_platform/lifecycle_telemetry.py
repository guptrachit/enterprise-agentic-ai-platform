import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger("agent_platform.lifecycle")


class LifecycleEventType(StrEnum):
    """Supported application lifecycle event types."""

    STARTUP_SUCCEEDED = "application_startup_succeeded"
    STARTUP_FAILED = "application_startup_failed"
    SHUTDOWN_STARTED = "application_shutdown_started"
    SHUTDOWN_COMPLETED = "application_shutdown_completed"


@dataclass(frozen=True)
class LifecycleEvent:
    """Safe structured application lifecycle telemetry."""

    timestamp: str
    event_type: LifecycleEventType
    app_env: str


def create_lifecycle_event(
    *,
    event_type: LifecycleEventType,
    app_env: str,
) -> LifecycleEvent:
    """Create one safe lifecycle telemetry event."""

    return LifecycleEvent(
        timestamp=datetime.now(UTC).isoformat(),
        event_type=event_type,
        app_env=app_env,
    )


def log_lifecycle_event(
    event: LifecycleEvent,
) -> None:
    """Log one structured lifecycle event."""

    logger.info(
        "lifecycle_event %s",
        json.dumps(
            asdict(event),
            sort_keys=True,
        ),
    )
