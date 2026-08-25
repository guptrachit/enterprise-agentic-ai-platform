import pytest

from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.routing_policy_comparison import (
    compare_routing_policies,
)
from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)
from agent_platform.llm.routing_policy_simulator import (
    RoutingPolicySimulator,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.routing_simulation import RoutingSimulationCase
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
    workload: LLMWorkload,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider="openai",
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                workload,
            }
        ),
    )


def create_simulator(
    *,
    models: dict[str, ModelDefinition],
    assignments: dict[
        LLMWorkload,
        str | tuple[str, ...],
    ],
    policy_name: str | None = None,
    policy_version: str | None = None,
) -> RoutingPolicySimulator:
    policy = ModelPolicy(assignments=assignments)

    router = ModelRouter(
        models=models,
        policy=policy,
    )

    versioned_policy = None

    if policy_name is not None and policy_version is not None:
        versioned_policy = VersionedRoutingPolicy(
            policy=policy,
            metadata=RoutingPolicyMetadata(
                name=policy_name,
                version=RoutingPolicyVersion.parse(policy_version),
            ),
        )

    return RoutingPolicySimulator(
        router,
        versioned_policy=versioned_policy,
    )


def test_policy_comparison_detects_no_changes() -> None:
    general = create_model(
        name="general",
        workload=LLMWorkload.GENERAL,
    )

    baseline = create_simulator(
        models={
            "general": general,
        },
        assignments={
            LLMWorkload.GENERAL: "general",
        },
    )

    candidate = create_simulator(
        models={
            "general": general,
        },
        assignments={
            LLMWorkload.GENERAL: "general",
        },
    )

    cases = (
        RoutingSimulationCase(
            name="general-case",
            workload=LLMWorkload.GENERAL,
        ),
    )

    comparison = compare_routing_policies(
        baseline=baseline,
        candidate=candidate,
        cases=cases,
    )

    assert comparison.baseline_policy_identifier is None
    assert comparison.candidate_policy_identifier is None
    assert comparison.total_cases == 1
    assert comparison.unchanged_cases == 1
    assert comparison.changed_count == 0
    assert comparison.changed_cases == ()
    assert comparison.change_rate == 0.0


def test_policy_comparison_detects_model_change() -> None:
    baseline_model = create_model(
        name="baseline_general",
        workload=LLMWorkload.GENERAL,
    )

    candidate_model = create_model(
        name="candidate_general",
        workload=LLMWorkload.GENERAL,
    )

    baseline = create_simulator(
        models={
            "baseline_general": baseline_model,
        },
        assignments={
            LLMWorkload.GENERAL: "baseline_general",
        },
    )

    candidate = create_simulator(
        models={
            "candidate_general": candidate_model,
        },
        assignments={
            LLMWorkload.GENERAL: "candidate_general",
        },
    )

    cases = (
        RoutingSimulationCase(
            name="general-case",
            workload=LLMWorkload.GENERAL,
        ),
    )

    comparison = compare_routing_policies(
        baseline=baseline,
        candidate=candidate,
        cases=cases,
    )

    assert comparison.total_cases == 1
    assert comparison.unchanged_cases == 0
    assert comparison.changed_count == 1
    assert comparison.change_rate == 1.0

    change = comparison.changed_cases[0]

    assert change.case_name == "general-case"
    assert change.baseline_model == "baseline_general"
    assert change.candidate_model == "candidate_general"


def test_policy_comparison_handles_mixed_changes() -> None:
    baseline_general = create_model(
        name="baseline_general",
        workload=LLMWorkload.GENERAL,
    )

    candidate_general = create_model(
        name="candidate_general",
        workload=LLMWorkload.GENERAL,
    )

    classification = create_model(
        name="classification",
        workload=LLMWorkload.CLASSIFICATION,
    )

    baseline = create_simulator(
        models={
            "baseline_general": baseline_general,
            "classification": classification,
        },
        assignments={
            LLMWorkload.GENERAL: "baseline_general",
            LLMWorkload.CLASSIFICATION: "classification",
        },
    )

    candidate = create_simulator(
        models={
            "candidate_general": candidate_general,
            "classification": classification,
        },
        assignments={
            LLMWorkload.GENERAL: "candidate_general",
            LLMWorkload.CLASSIFICATION: "classification",
        },
    )

    cases = (
        RoutingSimulationCase(
            name="general-case",
            workload=LLMWorkload.GENERAL,
        ),
        RoutingSimulationCase(
            name="classification-case",
            workload=LLMWorkload.CLASSIFICATION,
        ),
    )

    comparison = compare_routing_policies(
        baseline=baseline,
        candidate=candidate,
        cases=cases,
    )

    assert comparison.total_cases == 2
    assert comparison.unchanged_cases == 1
    assert comparison.changed_count == 1

    assert comparison.change_rate == pytest.approx(0.5)

    assert comparison.changed_cases[0].case_name == ("general-case")


def test_policy_comparison_preserves_change_order() -> None:
    baseline_general = create_model(
        name="baseline_general",
        workload=LLMWorkload.GENERAL,
    )

    candidate_general = create_model(
        name="candidate_general",
        workload=LLMWorkload.GENERAL,
    )

    baseline_classification = create_model(
        name="baseline_classification",
        workload=LLMWorkload.CLASSIFICATION,
    )

    candidate_classification = create_model(
        name="candidate_classification",
        workload=LLMWorkload.CLASSIFICATION,
    )

    baseline = create_simulator(
        models={
            "baseline_general": baseline_general,
            "baseline_classification": baseline_classification,
        },
        assignments={
            LLMWorkload.GENERAL: "baseline_general",
            LLMWorkload.CLASSIFICATION: "baseline_classification",
        },
    )

    candidate = create_simulator(
        models={
            "candidate_general": candidate_general,
            "candidate_classification": candidate_classification,
        },
        assignments={
            LLMWorkload.GENERAL: "candidate_general",
            LLMWorkload.CLASSIFICATION: "candidate_classification",
        },
    )

    cases = (
        RoutingSimulationCase(
            name="general-case",
            workload=LLMWorkload.GENERAL,
        ),
        RoutingSimulationCase(
            name="classification-case",
            workload=LLMWorkload.CLASSIFICATION,
        ),
    )

    comparison = compare_routing_policies(
        baseline=baseline,
        candidate=candidate,
        cases=cases,
    )

    assert tuple(change.case_name for change in comparison.changed_cases) == (
        "general-case",
        "classification-case",
    )


def test_policy_comparison_empty_case_set() -> None:
    general = create_model(
        name="general",
        workload=LLMWorkload.GENERAL,
    )

    simulator = create_simulator(
        models={
            "general": general,
        },
        assignments={
            LLMWorkload.GENERAL: "general",
        },
    )

    comparison = compare_routing_policies(
        baseline=simulator,
        candidate=simulator,
        cases=(),
    )

    assert comparison.total_cases == 0
    assert comparison.unchanged_cases == 0
    assert comparison.changed_count == 0
    assert comparison.change_rate == 0.0


def test_policy_comparison_includes_policy_identifiers() -> None:
    baseline_model = create_model(
        name="baseline_model",
        workload=LLMWorkload.GENERAL,
    )

    candidate_model = create_model(
        name="candidate_model",
        workload=LLMWorkload.GENERAL,
    )

    baseline = create_simulator(
        models={
            "baseline_model": baseline_model,
        },
        assignments={
            LLMWorkload.GENERAL: "baseline_model",
        },
        policy_name="production-routing-policy",
        policy_version="1.4.0",
    )

    candidate = create_simulator(
        models={
            "candidate_model": candidate_model,
        },
        assignments={
            LLMWorkload.GENERAL: "candidate_model",
        },
        policy_name="production-routing-policy",
        policy_version="1.5.0",
    )

    comparison = compare_routing_policies(
        baseline=baseline,
        candidate=candidate,
        cases=(
            RoutingSimulationCase(
                name="general-case",
                workload=LLMWorkload.GENERAL,
            ),
        ),
    )

    assert comparison.baseline_policy_identifier == ("production-routing-policy@1.4.0")

    assert comparison.candidate_policy_identifier == ("production-routing-policy@1.5.0")

    assert comparison.changed_count == 1
