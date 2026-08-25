import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)

logger = logging.getLogger("agent_platform.llm")


@dataclass(frozen=True)
class RuntimePolicyRefreshEvent:
    """Structured telemetry describing runtime policy refresh behavior."""

    timestamp: str
    policy_name: str
    refresh_mode: str
    policy_identifier: str
    refreshed: bool
    previous_policy_identifier: str | None


def create_runtime_policy_refresh_event(
    *,
    policy_name: str,
    refresh_mode: RuntimePolicyRefreshMode,
    policy_identifier: str,
    refreshed: bool,
    previous_policy_identifier: str | None,
) -> RuntimePolicyRefreshEvent:
    """Create structured runtime policy refresh telemetry."""

    return RuntimePolicyRefreshEvent(
        timestamp=datetime.now(UTC).isoformat(),
        policy_name=policy_name,
        refresh_mode=refresh_mode.value,
        policy_identifier=policy_identifier,
        refreshed=refreshed,
        previous_policy_identifier=previous_policy_identifier,
    )


def log_runtime_policy_refresh_event(
    event: RuntimePolicyRefreshEvent,
) -> None:
    """Log structured runtime policy refresh telemetry."""

    logger.info(
        "llm_runtime_policy_refresh %s",
        json.dumps(
            asdict(event),
            sort_keys=True,
        ),
    )
