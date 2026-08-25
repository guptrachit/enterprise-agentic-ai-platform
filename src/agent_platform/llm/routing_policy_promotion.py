import json
from dataclasses import dataclass
from enum import StrEnum

from agent_platform.llm.routing_deployment_gate import (
    RoutingDeploymentGateResult,
)
from agent_platform.llm.routing_policy_transition import (
    RoutingPolicyTransitionResult,
)


class RoutingPolicyPromotionStatus(StrEnum):
    """Promotion eligibility status for a routing policy."""

    ALLOWED = "allowed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RoutingPolicyPromotionDecision:
    """Final governance decision for routing policy promotion."""

    status: RoutingPolicyPromotionStatus
    transition: RoutingPolicyTransitionResult
    deployment_gate: RoutingDeploymentGateResult
    reasons: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Return whether routing policy promotion is allowed."""

        return self.status is RoutingPolicyPromotionStatus.ALLOWED

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable promotion report."""

        return {
            "status": self.status.value,
            "allowed": self.allowed,
            "baseline_version": str(self.transition.baseline_version),
            "candidate_version": str(self.transition.candidate_version),
            "transition_status": (self.transition.status.value),
            "transition_reason": (self.transition.reason),
            "policy_identifier": (self.deployment_gate.policy_identifier),
            "deployment_gate_status": (self.deployment_gate.status.value),
            "minimum_expectation_pass_rate": (
                self.deployment_gate.minimum_expectation_pass_rate
            ),
            "actual_expectation_pass_rate": (
                self.deployment_gate.actual_expectation_pass_rate
            ),
            "failed_expectations": (self.deployment_gate.failed_expectations),
            "failed_case_names": list(self.deployment_gate.failed_case_names),
            "reasons": list(self.reasons),
        }

    def to_json(self) -> str:
        """Return the promotion decision as stable JSON."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
        )


def evaluate_routing_policy_promotion(
    *,
    transition: RoutingPolicyTransitionResult,
    deployment_gate: RoutingDeploymentGateResult,
) -> RoutingPolicyPromotionDecision:
    """Evaluate whether a routing policy is eligible for promotion."""

    reasons: list[str] = []

    if not transition.allowed:
        reasons.append(f"Version transition rejected: {transition.reason}")

    if not deployment_gate.passed:
        reasons.append("Routing policy deployment gate failed.")

    status = (
        RoutingPolicyPromotionStatus.ALLOWED
        if transition.allowed and deployment_gate.passed
        else RoutingPolicyPromotionStatus.REJECTED
    )

    if status is RoutingPolicyPromotionStatus.ALLOWED:
        reasons.append(
            "Routing policy version transition and deployment gate both passed."
        )

    return RoutingPolicyPromotionDecision(
        status=status,
        transition=transition,
        deployment_gate=deployment_gate,
        reasons=tuple(reasons),
    )
