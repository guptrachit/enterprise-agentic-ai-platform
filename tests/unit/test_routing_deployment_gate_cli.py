from agent_platform.llm.routing_deployment_gate import (
    RoutingDeploymentGateResult,
    RoutingDeploymentGateStatus,
)
from agent_platform.llm.routing_deployment_gate_cli import (
    deployment_gate_exit_code,
)


def create_result(
    *,
    status: RoutingDeploymentGateStatus,
) -> RoutingDeploymentGateResult:
    return RoutingDeploymentGateResult(
        policy_identifier=None,
        status=status,
        minimum_expectation_pass_rate=1.0,
        actual_expectation_pass_rate=(
            1.0 if status is RoutingDeploymentGateStatus.PASSED else 0.9
        ),
        total_cases=10,
        cases_with_expectations=10,
        failed_expectations=(0 if status is RoutingDeploymentGateStatus.PASSED else 1),
        failed_case_names=(
            () if status is RoutingDeploymentGateStatus.PASSED else ("failed-case",)
        ),
    )


def test_deployment_gate_exit_code_is_zero_when_passed() -> None:
    result = create_result(
        status=RoutingDeploymentGateStatus.PASSED,
    )

    assert deployment_gate_exit_code(result) == 0


def test_deployment_gate_exit_code_is_one_when_failed() -> None:
    result = create_result(
        status=RoutingDeploymentGateStatus.FAILED,
    )

    assert deployment_gate_exit_code(result) == 1
