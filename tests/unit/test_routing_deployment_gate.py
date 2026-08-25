import json

import pytest

from agent_platform.llm.routing_deployment_gate import (
    RoutingDeploymentGateStatus,
    evaluate_routing_deployment_gate,
)
from agent_platform.llm.routing_policy_simulator import (
    RoutingSimulationSummary,
)


def create_summary(
    *,
    total_cases: int,
    cases_with_expectations: int,
    passed_expectations: int,
    failed_expectations: int,
    failed_case_names: tuple[str, ...] = (),
    policy_identifier: str | None = None,
) -> RoutingSimulationSummary:
    return RoutingSimulationSummary(
        policy_identifier=policy_identifier,
        total_cases=total_cases,
        cases_with_expectations=cases_with_expectations,
        passed_expectations=passed_expectations,
        failed_expectations=failed_expectations,
        failed_case_names=failed_case_names,
    )


def test_deployment_gate_passes_when_all_expectations_pass() -> None:
    summary = create_summary(
        total_cases=10,
        cases_with_expectations=10,
        passed_expectations=10,
        failed_expectations=0,
    )

    result = evaluate_routing_deployment_gate(summary)

    assert result.policy_identifier is None
    assert result.status is RoutingDeploymentGateStatus.PASSED
    assert result.passed is True
    assert result.minimum_expectation_pass_rate == 1.0
    assert result.actual_expectation_pass_rate == 1.0
    assert result.failed_expectations == 0
    assert result.failed_case_names == ()


def test_deployment_gate_fails_when_default_threshold_not_met() -> None:
    summary = create_summary(
        total_cases=10,
        cases_with_expectations=10,
        passed_expectations=9,
        failed_expectations=1,
        failed_case_names=("classification-case",),
    )

    result = evaluate_routing_deployment_gate(summary)

    assert result.status is RoutingDeploymentGateStatus.FAILED
    assert result.passed is False

    assert result.actual_expectation_pass_rate == pytest.approx(0.9)

    assert result.failed_case_names == ("classification-case",)


def test_deployment_gate_supports_custom_threshold() -> None:
    summary = create_summary(
        total_cases=20,
        cases_with_expectations=20,
        passed_expectations=19,
        failed_expectations=1,
        failed_case_names=("one-failed-case",),
    )

    result = evaluate_routing_deployment_gate(
        summary,
        minimum_expectation_pass_rate=0.95,
    )

    assert result.status is RoutingDeploymentGateStatus.PASSED
    assert result.passed is True

    assert result.actual_expectation_pass_rate == pytest.approx(0.95)


def test_deployment_gate_fails_below_custom_threshold() -> None:
    summary = create_summary(
        total_cases=20,
        cases_with_expectations=20,
        passed_expectations=18,
        failed_expectations=2,
        failed_case_names=(
            "case-a",
            "case-b",
        ),
    )

    result = evaluate_routing_deployment_gate(
        summary,
        minimum_expectation_pass_rate=0.95,
    )

    assert result.status is RoutingDeploymentGateStatus.FAILED
    assert result.passed is False

    assert result.actual_expectation_pass_rate == pytest.approx(0.9)


def test_deployment_gate_passes_when_no_expectations_exist() -> None:
    summary = create_summary(
        total_cases=5,
        cases_with_expectations=0,
        passed_expectations=0,
        failed_expectations=0,
    )

    result = evaluate_routing_deployment_gate(summary)

    assert result.status is RoutingDeploymentGateStatus.PASSED
    assert result.passed is True
    assert result.actual_expectation_pass_rate == 1.0


@pytest.mark.parametrize(
    "threshold",
    (
        -0.01,
        1.01,
        -1.0,
        2.0,
    ),
)
def test_deployment_gate_rejects_invalid_threshold(
    threshold: float,
) -> None:
    summary = create_summary(
        total_cases=1,
        cases_with_expectations=1,
        passed_expectations=1,
        failed_expectations=0,
    )

    with pytest.raises(
        ValueError,
        match=("minimum_expectation_pass_rate must be between 0.0 and 1.0"),
    ):
        evaluate_routing_deployment_gate(
            summary,
            minimum_expectation_pass_rate=threshold,
        )


def test_deployment_gate_preserves_summary_context() -> None:
    summary = create_summary(
        total_cases=12,
        cases_with_expectations=10,
        passed_expectations=8,
        failed_expectations=2,
        failed_case_names=(
            "tool-case",
            "cost-case",
        ),
    )

    result = evaluate_routing_deployment_gate(
        summary,
        minimum_expectation_pass_rate=0.9,
    )

    assert result.total_cases == 12
    assert result.cases_with_expectations == 10
    assert result.failed_expectations == 2

    assert result.failed_case_names == (
        "tool-case",
        "cost-case",
    )


def test_deployment_gate_preserves_policy_identifier() -> None:
    summary = create_summary(
        total_cases=10,
        cases_with_expectations=10,
        passed_expectations=10,
        failed_expectations=0,
        policy_identifier=("production-routing-policy@1.5.0"),
    )

    result = evaluate_routing_deployment_gate(
        summary,
        minimum_expectation_pass_rate=0.95,
    )

    assert result.policy_identifier == ("production-routing-policy@1.5.0")

    assert result.status is RoutingDeploymentGateStatus.PASSED


def test_deployment_gate_to_dict() -> None:
    summary = create_summary(
        total_cases=10,
        cases_with_expectations=10,
        passed_expectations=9,
        failed_expectations=1,
        failed_case_names=("classification-case",),
        policy_identifier=("candidate-routing-policy@2.0.0"),
    )

    result = evaluate_routing_deployment_gate(
        summary,
        minimum_expectation_pass_rate=0.95,
    )

    payload = result.to_dict()

    assert payload == {
        "policy_identifier": ("candidate-routing-policy@2.0.0"),
        "status": "failed",
        "passed": False,
        "minimum_expectation_pass_rate": 0.95,
        "actual_expectation_pass_rate": 0.9,
        "total_cases": 10,
        "cases_with_expectations": 10,
        "failed_expectations": 1,
        "failed_case_names": [
            "classification-case",
        ],
    }


def test_deployment_gate_to_dict_is_independent() -> None:
    summary = create_summary(
        total_cases=2,
        cases_with_expectations=2,
        passed_expectations=1,
        failed_expectations=1,
        failed_case_names=("failed-case",),
    )

    result = evaluate_routing_deployment_gate(summary)

    payload = result.to_dict()

    failed_case_names = payload["failed_case_names"]

    assert isinstance(
        failed_case_names,
        list,
    )

    failed_case_names.append("mutated-case")

    assert result.failed_case_names == ("failed-case",)


def test_deployment_gate_to_json() -> None:
    summary = create_summary(
        total_cases=4,
        cases_with_expectations=4,
        passed_expectations=4,
        failed_expectations=0,
        policy_identifier=("production-routing-policy@3.1.0"),
    )

    result = evaluate_routing_deployment_gate(
        summary,
        minimum_expectation_pass_rate=0.95,
    )

    serialized = result.to_json()

    payload = json.loads(serialized)

    assert payload["policy_identifier"] == ("production-routing-policy@3.1.0")

    assert payload["status"] == "passed"
    assert payload["passed"] is True

    assert payload["minimum_expectation_pass_rate"] == 0.95

    assert payload["actual_expectation_pass_rate"] == 1.0

    assert payload["failed_case_names"] == []


def test_deployment_gate_json_is_stable() -> None:
    summary = create_summary(
        total_cases=1,
        cases_with_expectations=1,
        passed_expectations=1,
        failed_expectations=0,
        policy_identifier="policy@1.0.0",
    )

    result = evaluate_routing_deployment_gate(summary)

    first = result.to_json()
    second = result.to_json()

    assert first == second
