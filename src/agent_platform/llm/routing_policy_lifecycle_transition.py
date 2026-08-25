from dataclasses import dataclass
from enum import StrEnum

from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)


class RoutingPolicyLifecycleTransitionStatus(StrEnum):
    """Validation result for a routing policy lifecycle transition."""

    ALLOWED = "allowed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RoutingPolicyLifecycleTransitionResult:
    """Result of validating a routing policy lifecycle transition."""

    current_status: RoutingPolicyLifecycleStatus
    target_status: RoutingPolicyLifecycleStatus
    status: RoutingPolicyLifecycleTransitionStatus
    reason: str

    @property
    def allowed(self) -> bool:
        """Return whether the lifecycle transition is allowed."""

        return self.status is RoutingPolicyLifecycleTransitionStatus.ALLOWED


_ALLOWED_TRANSITIONS: dict[
    RoutingPolicyLifecycleStatus,
    frozenset[RoutingPolicyLifecycleStatus],
] = {
    RoutingPolicyLifecycleStatus.DRAFT: frozenset(
        {
            RoutingPolicyLifecycleStatus.CANDIDATE,
        }
    ),
    RoutingPolicyLifecycleStatus.CANDIDATE: frozenset(
        {
            RoutingPolicyLifecycleStatus.APPROVED,
        }
    ),
    RoutingPolicyLifecycleStatus.APPROVED: frozenset(
        {
            RoutingPolicyLifecycleStatus.ACTIVE,
        }
    ),
    RoutingPolicyLifecycleStatus.ACTIVE: frozenset(
        {
            RoutingPolicyLifecycleStatus.RETIRED,
        }
    ),
    RoutingPolicyLifecycleStatus.RETIRED: frozenset(),
}


def evaluate_lifecycle_transition(
    *,
    current_status: RoutingPolicyLifecycleStatus,
    target_status: RoutingPolicyLifecycleStatus,
) -> RoutingPolicyLifecycleTransitionResult:
    """Validate a routing policy lifecycle transition."""

    if target_status in _ALLOWED_TRANSITIONS[current_status]:
        return RoutingPolicyLifecycleTransitionResult(
            current_status=current_status,
            target_status=target_status,
            status=(RoutingPolicyLifecycleTransitionStatus.ALLOWED),
            reason=(
                f"Lifecycle transition from "
                f"'{current_status.value}' to "
                f"'{target_status.value}' is allowed."
            ),
        )

    return RoutingPolicyLifecycleTransitionResult(
        current_status=current_status,
        target_status=target_status,
        status=(RoutingPolicyLifecycleTransitionStatus.REJECTED),
        reason=(
            f"Lifecycle transition from "
            f"'{current_status.value}' to "
            f"'{target_status.value}' is not allowed."
        ),
    )
