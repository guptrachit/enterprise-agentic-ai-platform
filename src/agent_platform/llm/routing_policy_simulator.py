from dataclasses import dataclass

from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.routing_simulation import (
    RoutingSimulationCase,
    RoutingSimulationResult,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)


@dataclass(frozen=True)
class RoutingSimulationSummary:
    """Aggregate expectation results for a routing simulation suite."""

    policy_identifier: str | None
    total_cases: int
    cases_with_expectations: int
    passed_expectations: int
    failed_expectations: int
    failed_case_names: tuple[str, ...]

    @property
    def expectation_pass_rate(self) -> float:
        """Return the fraction of declared expectations that passed."""

        if self.cases_with_expectations == 0:
            return 1.0

        return self.passed_expectations / self.cases_with_expectations

    @property
    def passed(self) -> bool:
        """Return whether all declared expectations passed."""

        return self.failed_expectations == 0


class RoutingPolicySimulator:
    """Simulates model-routing decisions without provider execution."""

    def __init__(
        self,
        router: ModelRouter,
        versioned_policy: VersionedRoutingPolicy | None = None,
    ) -> None:
        self.router = router
        self.versioned_policy = versioned_policy

    @property
    def policy_identifier(self) -> str | None:
        """Return the governed policy identifier when available."""

        if self.versioned_policy is None:
            return None

        return self.versioned_policy.identifier

    def simulate(
        self,
        case: RoutingSimulationCase,
    ) -> RoutingSimulationResult:
        """Simulate one routing case."""

        decision = self.router.route_decision(
            case.workload,
            constraints=case.constraints,
            required_capabilities=case.required_capabilities,
            preference=case.preference,
        )

        return RoutingSimulationResult(
            case=case,
            decision=decision,
        )

    def simulate_many(
        self,
        cases: tuple[RoutingSimulationCase, ...],
    ) -> tuple[RoutingSimulationResult, ...]:
        """Simulate multiple routing cases in input order."""

        return tuple(self.simulate(case) for case in cases)

    def expectation_failures(
        self,
        cases: tuple[RoutingSimulationCase, ...],
    ) -> tuple[RoutingSimulationResult, ...]:
        """Return simulation results that fail declared expectations."""

        return tuple(
            result
            for result in self.simulate_many(cases)
            if result.matches_expectation is False
        )

    def summarize(
        self,
        cases: tuple[RoutingSimulationCase, ...],
    ) -> RoutingSimulationSummary:
        """Summarize expectation outcomes for a simulation suite."""

        results = self.simulate_many(cases)

        evaluated = tuple(
            result for result in results if result.matches_expectation is not None
        )

        failures = tuple(
            result for result in evaluated if result.matches_expectation is False
        )

        passed_expectations = sum(
            result.matches_expectation is True for result in evaluated
        )

        return RoutingSimulationSummary(
            policy_identifier=self.policy_identifier,
            total_cases=len(results),
            cases_with_expectations=len(evaluated),
            passed_expectations=passed_expectations,
            failed_expectations=len(failures),
            failed_case_names=tuple(result.case.name for result in failures),
        )
