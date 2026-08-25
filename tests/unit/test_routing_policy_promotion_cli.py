from agent_platform.llm.routing_deployment_gate import (
    RoutingDeploymentGateResult,
    RoutingDeploymentGateStatus,
)
from agent_platform.llm.routing_policy_promotion import (
    RoutingPolicyPromotionDecision,
    RoutingPolicyPromotionStatus,
)
from agent_platform.llm.routing_policy_promotion_cli import (
    promotion_exit_code,
)
from agent_platform.llm.routing_policy_transition import (
    RoutingPolicyTransitionResult,
    RoutingPolicyTransitionStatus,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)


def version(
    value: str,
) -> RoutingPolicyVersion:
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


def create_decision(
    *,
    allowed: bool,
) -> RoutingPolicyPromotionDecision:
    transition = create_transition(
        allowed=allowed,
    )

    gate = create_gate(
        passed=allowed,
    )

    return RoutingPolicyPromotionDecision(
        status=(
            RoutingPolicyPromotionStatus.ALLOWED
            if allowed
            else RoutingPolicyPromotionStatus.REJECTED
        ),
        transition=transition,
        deployment_gate=gate,
        reasons=(
            ("Routing policy version transition and deployment gate both passed."),
        )
        if allowed
        else ("Routing policy promotion rejected.",),
    )


def test_promotion_exit_code_is_zero_when_allowed() -> None:
    decision = create_decision(
        allowed=True,
    )

    assert promotion_exit_code(decision) == 0


def test_promotion_exit_code_is_one_when_rejected() -> None:
    decision = create_decision(
        allowed=False,
    )

    assert promotion_exit_code(decision) == 1
