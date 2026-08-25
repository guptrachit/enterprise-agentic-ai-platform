from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)


class RoutingPolicyActivationResultLike(Protocol):
    """Structural contract for activation results used by audit creation."""

    activated_policy: GovernedRoutingPolicy
    retired_policy: GovernedRoutingPolicy | None


@dataclass(frozen=True)
class RoutingPolicyActivationAuditEvent:
    """Immutable audit event for a routing policy activation."""

    timestamp: str
    policy_name: str
    activated_policy_identifier: str
    activated_version: str
    retired_policy_identifier: str | None
    retired_version: str | None


def create_activation_audit_event(
    result: RoutingPolicyActivationResultLike,
) -> RoutingPolicyActivationAuditEvent:
    """Create an immutable audit event from an activation result."""

    activated = result.activated_policy
    retired = result.retired_policy

    return RoutingPolicyActivationAuditEvent(
        timestamp=datetime.now(UTC).isoformat(),
        policy_name=activated.policy_name,
        activated_policy_identifier=activated.identifier,
        activated_version=activated.version,
        retired_policy_identifier=(retired.identifier if retired is not None else None),
        retired_version=(retired.version if retired is not None else None),
    )
