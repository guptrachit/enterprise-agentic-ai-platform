from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.routing_decision import RoutingDecision
from agent_platform.llm.routing_reason import (
    RoutingReason,
    RoutingReasonCode,
)
from agent_platform.llm.routing_simulation import (
    RoutingSimulationCase,
    RoutingSimulationResult,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(
    name: str,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider="openai",
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )


def create_result(
    *,
    selected_model: str,
    expected_selected_model: str | None,
) -> RoutingSimulationResult:
    model = create_model(selected_model)

    decision = RoutingDecision(
        selected_model=model,
        ranked_candidates=(model,),
        reasons=(
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message=f"Selected model '{selected_model}'",
            ),
        ),
    )

    case = RoutingSimulationCase(
        name="general-routing-case",
        workload=LLMWorkload.GENERAL,
        expected_selected_model=expected_selected_model,
    )

    return RoutingSimulationResult(
        case=case,
        decision=decision,
    )


def test_routing_simulation_case_preserves_input() -> None:
    case = RoutingSimulationCase(
        name="general-routing-case",
        workload=LLMWorkload.GENERAL,
        expected_selected_model="primary",
    )

    assert case.name == "general-routing-case"
    assert case.workload is LLMWorkload.GENERAL
    assert case.expected_selected_model == "primary"


def test_routing_simulation_result_preserves_decision() -> None:
    result = create_result(
        selected_model="primary",
        expected_selected_model="primary",
    )

    assert result.selected_model.name == "primary"
    assert result.decision.selected_model is result.selected_model


def test_simulation_matches_expected_model() -> None:
    result = create_result(
        selected_model="primary",
        expected_selected_model="primary",
    )

    assert result.matches_expectation is True


def test_simulation_detects_unexpected_model() -> None:
    result = create_result(
        selected_model="backup",
        expected_selected_model="primary",
    )

    assert result.matches_expectation is False


def test_simulation_without_expectation_returns_none() -> None:
    result = create_result(
        selected_model="primary",
        expected_selected_model=None,
    )

    assert result.matches_expectation is None
