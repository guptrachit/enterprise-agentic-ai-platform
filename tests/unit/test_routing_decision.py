from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.routing_decision import RoutingDecision
from agent_platform.llm.routing_reason import (
    RoutingReason,
    RoutingReasonCode,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(name: str) -> ModelDefinition:
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


def test_routing_decision_preserves_selected_model() -> None:
    selected = create_model("selected")

    decision = RoutingDecision(
        selected_model=selected,
        ranked_candidates=(selected,),
    )

    assert decision.selected_model is selected
    assert decision.ranked_candidates == (selected,)
    assert decision.rejected_models == ()
    assert decision.reasons == ()


def test_routing_decision_preserves_structured_reasons() -> None:
    selected = create_model("cheap_model")
    alternative = create_model("expensive_model")

    reasons = (
        RoutingReason(
            code=RoutingReasonCode.LOWER_COST_PREFERRED,
            message=("cheap_model ranked first because lower cost was preferred"),
        ),
        RoutingReason(
            code=RoutingReasonCode.DISABLED,
            message=("disabled_model rejected because it is disabled"),
        ),
    )

    decision = RoutingDecision(
        selected_model=selected,
        ranked_candidates=(
            selected,
            alternative,
        ),
        rejected_models=("disabled_model",),
        reasons=reasons,
    )

    assert decision.selected_model.name == "cheap_model"

    assert tuple(model.name for model in decision.ranked_candidates) == (
        "cheap_model",
        "expensive_model",
    )

    assert decision.rejected_models == ("disabled_model",)

    assert decision.reasons == reasons

    assert decision.reasons[0].code is RoutingReasonCode.LOWER_COST_PREFERRED
