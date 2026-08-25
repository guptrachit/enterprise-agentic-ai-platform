from dataclasses import dataclass

from agent_platform.llm.routing_policy_simulator import (
    RoutingPolicySimulator,
)
from agent_platform.llm.routing_simulation import RoutingSimulationCase


@dataclass(frozen=True)
class RoutingPolicyChange:
    """One model-selection change between two routing policies."""

    case_name: str
    baseline_model: str
    candidate_model: str


@dataclass(frozen=True)
class RoutingPolicyComparison:
    """Comparison result for two routing policies."""

    baseline_policy_identifier: str | None
    candidate_policy_identifier: str | None
    total_cases: int
    unchanged_cases: int
    changed_cases: tuple[RoutingPolicyChange, ...]

    @property
    def changed_count(self) -> int:
        """Return the number of cases with changed model selection."""

        return len(self.changed_cases)

    @property
    def change_rate(self) -> float:
        """Return the fraction of cases whose model selection changed."""

        if self.total_cases == 0:
            return 0.0

        return self.changed_count / self.total_cases


def compare_routing_policies(
    *,
    baseline: RoutingPolicySimulator,
    candidate: RoutingPolicySimulator,
    cases: tuple[RoutingSimulationCase, ...],
) -> RoutingPolicyComparison:
    """Compare model selections for the same cases across two policies."""

    baseline_results = baseline.simulate_many(cases)
    candidate_results = candidate.simulate_many(cases)

    changes: list[RoutingPolicyChange] = []
    unchanged_cases = 0

    for baseline_result, candidate_result in zip(
        baseline_results,
        candidate_results,
        strict=True,
    ):
        baseline_model = baseline_result.selected_model.name
        candidate_model = candidate_result.selected_model.name

        if baseline_model == candidate_model:
            unchanged_cases += 1
            continue

        changes.append(
            RoutingPolicyChange(
                case_name=baseline_result.case.name,
                baseline_model=baseline_model,
                candidate_model=candidate_model,
            )
        )

    return RoutingPolicyComparison(
        baseline_policy_identifier=baseline.policy_identifier,
        candidate_policy_identifier=candidate.policy_identifier,
        total_cases=len(cases),
        unchanged_cases=unchanged_cases,
        changed_cases=tuple(changes),
    )
