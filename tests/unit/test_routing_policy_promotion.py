import json

from agent_platform.llm.routing_deployment_gate import (
    RoutingDeploymentGateResult,
    RoutingDeploymentGateStatus,
)
from agent_platform.llm.routing_policy_promotion import (
    RoutingPolicyPromotionStatus,
    evaluate_routing_policy_promotion,
)
from agent_platform.llm.routing_policy_transition import (
    RoutingPolicyTransitionResult,
    RoutingPolicyTransitionStatus,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)


def version(value: str) -> RoutingPolicyVersion:
    return RoutingPolicyVersion.parse(value)


def create_transition(
    *,
    allowed: bool,
) -> RoutingPolicyTransitionResult:
    return RoutingPolicyTransitionResult(
        baseline_version=version("1.4.0"),
        candidate_version=(version("1.5.0") if allowed else version("1.4.0")),
        status=(
            RoutingPolicyTransitionStatus.ALLOWED
            if allowed
            else RoutingPolicyTransitionStatus.REJECTED
        ),
        reason=(
            "Candidate routing policy version is newer than the baseline version."
            if allowed
            else (
                "Candidate routing policy version must be newer "
                "than the baseline version."
            )
        ),
    )


def create_gate(
    *,
    passed: bool,
) -> RoutingDeploymentGateResult:
    return RoutingDeploymentGateResult(
        policy_identifier="production-routing-policy@1.5.0",
        status=(
            RoutingDeploymentGateStatus.PASSED
            if passed
            else RoutingDeploymentGateStatus.FAILED
        ),
        minimum_expectation_pass_rate=1.0,
        actual_expectation_pass_rate=(1.0 if passed else 0.9),
        total_cases=10,
        cases_with_expectations=10,
        failed_expectations=(0 if passed else 1),
        failed_case_names=(() if passed else ("failed-case",)),
    )


def test_promotion_allowed_when_transition_and_gate_pass() -> None:
    transition = create_transition(
        allowed=True,
    )

    gate = create_gate(
        passed=True,
    )

    decision = evaluate_routing_policy_promotion(
        transition=transition,
        deployment_gate=gate,
    )

    assert decision.status is RoutingPolicyPromotionStatus.ALLOWED
    assert decision.allowed is True
    assert decision.transition is transition
    assert decision.deployment_gate is gate

    assert decision.reasons == (
        "Routing policy version transition and deployment gate both passed.",
    )


def test_promotion_rejected_when_transition_fails() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=False,
        ),
        deployment_gate=create_gate(
            passed=True,
        ),
    )

    assert decision.status is RoutingPolicyPromotionStatus.REJECTED
    assert decision.allowed is False

    assert len(decision.reasons) == 1

    assert decision.reasons[0].startswith("Version transition rejected:")


def test_promotion_rejected_when_deployment_gate_fails() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=True,
        ),
        deployment_gate=create_gate(
            passed=False,
        ),
    )

    assert decision.status is RoutingPolicyPromotionStatus.REJECTED
    assert decision.allowed is False

    assert decision.reasons == ("Routing policy deployment gate failed.",)


def test_promotion_rejected_when_both_checks_fail() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=False,
        ),
        deployment_gate=create_gate(
            passed=False,
        ),
    )

    assert decision.status is RoutingPolicyPromotionStatus.REJECTED
    assert decision.allowed is False

    assert len(decision.reasons) == 2

    assert decision.reasons[0].startswith("Version transition rejected:")

    assert decision.reasons[1] == ("Routing policy deployment gate failed.")


def test_promotion_preserves_governance_inputs() -> None:
    transition = create_transition(
        allowed=True,
    )

    gate = create_gate(
        passed=True,
    )

    decision = evaluate_routing_policy_promotion(
        transition=transition,
        deployment_gate=gate,
    )

    assert decision.transition.baseline_version == (version("1.4.0"))

    assert decision.transition.candidate_version == (version("1.5.0"))

    assert decision.deployment_gate.policy_identifier == (
        "production-routing-policy@1.5.0"
    )


def test_promotion_to_dict() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=True,
        ),
        deployment_gate=create_gate(
            passed=True,
        ),
    )

    payload = decision.to_dict()

    assert payload == {
        "status": "allowed",
        "allowed": True,
        "baseline_version": "1.4.0",
        "candidate_version": "1.5.0",
        "transition_status": "allowed",
        "transition_reason": (
            "Candidate routing policy version is newer than the baseline version."
        ),
        "policy_identifier": ("production-routing-policy@1.5.0"),
        "deployment_gate_status": "passed",
        "minimum_expectation_pass_rate": 1.0,
        "actual_expectation_pass_rate": 1.0,
        "failed_expectations": 0,
        "failed_case_names": [],
        "reasons": [
            "Routing policy version transition and deployment gate both passed."
        ],
    }


def test_rejected_promotion_to_dict() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=True,
        ),
        deployment_gate=create_gate(
            passed=False,
        ),
    )

    payload = decision.to_dict()

    assert payload["status"] == "rejected"
    assert payload["allowed"] is False
    assert payload["deployment_gate_status"] == "failed"
    assert payload["actual_expectation_pass_rate"] == 0.9
    assert payload["failed_expectations"] == 1

    assert payload["failed_case_names"] == [
        "failed-case",
    ]


def test_promotion_to_dict_is_independent() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=True,
        ),
        deployment_gate=create_gate(
            passed=False,
        ),
    )

    payload = decision.to_dict()

    failed_case_names = payload["failed_case_names"]
    reasons = payload["reasons"]

    assert isinstance(
        failed_case_names,
        list,
    )

    assert isinstance(
        reasons,
        list,
    )

    failed_case_names.append("mutated-case")

    reasons.append("mutated-reason")

    assert decision.deployment_gate.failed_case_names == ("failed-case",)

    assert decision.reasons == ("Routing policy deployment gate failed.",)


def test_promotion_to_json() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=True,
        ),
        deployment_gate=create_gate(
            passed=True,
        ),
    )

    serialized = decision.to_json()

    payload = json.loads(serialized)

    assert payload["status"] == "allowed"
    assert payload["allowed"] is True
    assert payload["baseline_version"] == "1.4.0"
    assert payload["candidate_version"] == "1.5.0"

    assert payload["policy_identifier"] == ("production-routing-policy@1.5.0")


def test_promotion_json_is_stable() -> None:
    decision = evaluate_routing_policy_promotion(
        transition=create_transition(
            allowed=True,
        ),
        deployment_gate=create_gate(
            passed=True,
        ),
    )

    first = decision.to_json()
    second = decision.to_json()

    assert first == second
