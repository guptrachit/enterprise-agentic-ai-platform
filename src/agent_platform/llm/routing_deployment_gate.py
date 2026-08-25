import json
from dataclasses import dataclass
from enum import StrEnum

from agent_platform.llm.routing_policy_simulator import (
    RoutingSimulationSummary,
)


class RoutingDeploymentGateStatus(StrEnum):
    """Deployment eligibility status for a routing policy."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class RoutingDeploymentGateResult:
    """Result of evaluating a routing simulation summary."""

    policy_identifier: str | None
    status: RoutingDeploymentGateStatus
    minimum_expectation_pass_rate: float
    actual_expectation_pass_rate: float
    total_cases: int
    cases_with_expectations: int
    failed_expectations: int
    failed_case_names: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the deployment gate passed."""

        return self.status is RoutingDeploymentGateStatus.PASSED

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable deployment gate report."""

        return {
            "policy_identifier": self.policy_identifier,
            "status": self.status.value,
            "passed": self.passed,
            "minimum_expectation_pass_rate": (self.minimum_expectation_pass_rate),
            "actual_expectation_pass_rate": (self.actual_expectation_pass_rate),
            "total_cases": self.total_cases,
            "cases_with_expectations": self.cases_with_expectations,
            "failed_expectations": self.failed_expectations,
            "failed_case_names": list(self.failed_case_names),
        }

    def to_json(self) -> str:
        """Return the deployment gate report as stable JSON."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
        )


def evaluate_routing_deployment_gate(
    summary: RoutingSimulationSummary,
    *,
    minimum_expectation_pass_rate: float = 1.0,
) -> RoutingDeploymentGateResult:
    """Evaluate whether a routing policy satisfies the deployment gate."""

    if not 0.0 <= minimum_expectation_pass_rate <= 1.0:
        raise ValueError("minimum_expectation_pass_rate must be between 0.0 and 1.0")

    actual_rate = summary.expectation_pass_rate

    status = (
        RoutingDeploymentGateStatus.PASSED
        if actual_rate >= minimum_expectation_pass_rate
        else RoutingDeploymentGateStatus.FAILED
    )

    return RoutingDeploymentGateResult(
        policy_identifier=summary.policy_identifier,
        status=status,
        minimum_expectation_pass_rate=minimum_expectation_pass_rate,
        actual_expectation_pass_rate=actual_rate,
        total_cases=summary.total_cases,
        cases_with_expectations=summary.cases_with_expectations,
        failed_expectations=summary.failed_expectations,
        failed_case_names=summary.failed_case_names,
    )
