from dataclasses import dataclass
from enum import StrEnum

from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)


class RoutingPolicyTransitionStatus(StrEnum):
    """Validation status for a routing policy version transition."""

    ALLOWED = "allowed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RoutingPolicyTransitionResult:
    """Result of validating a routing policy version transition."""

    baseline_version: RoutingPolicyVersion
    candidate_version: RoutingPolicyVersion
    status: RoutingPolicyTransitionStatus
    reason: str

    @property
    def allowed(self) -> bool:
        """Return whether the version transition is allowed."""

        return self.status is RoutingPolicyTransitionStatus.ALLOWED


def evaluate_routing_policy_transition(
    *,
    baseline_version: RoutingPolicyVersion,
    candidate_version: RoutingPolicyVersion,
) -> RoutingPolicyTransitionResult:
    """Validate that a candidate routing policy version is newer."""

    if candidate_version > baseline_version:
        return RoutingPolicyTransitionResult(
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            status=RoutingPolicyTransitionStatus.ALLOWED,
            reason=(
                "Candidate routing policy version is newer than the baseline version."
            ),
        )

    if candidate_version == baseline_version:
        reason = (
            "Candidate routing policy version must be newer than the baseline version."
        )
    else:
        reason = "Candidate routing policy version is older than the baseline version."

    return RoutingPolicyTransitionResult(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        status=RoutingPolicyTransitionStatus.REJECTED,
        reason=reason,
    )
