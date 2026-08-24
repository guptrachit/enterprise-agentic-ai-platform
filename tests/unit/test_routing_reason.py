from agent_platform.llm.routing_reason import (
    RoutingReason,
    RoutingReasonCode,
)


def test_routing_reason_preserves_code_and_message() -> None:
    reason = RoutingReason(
        code=RoutingReasonCode.CAPABILITY_MISSING,
        message=(
            "Model 'text_only' rejected because "
            "required capability 'tool_calling' is missing"
        ),
    )

    assert reason.code is RoutingReasonCode.CAPABILITY_MISSING

    assert reason.message == (
        "Model 'text_only' rejected because "
        "required capability 'tool_calling' is missing"
    )


def test_routing_reason_code_values_are_stable() -> None:
    assert RoutingReasonCode.DISABLED.value == "disabled"

    assert RoutingReasonCode.WORKLOAD_NOT_SUPPORTED.value == "workload_not_supported"

    assert RoutingReasonCode.CONSTRAINT_REJECTED.value == "constraint_rejected"

    assert RoutingReasonCode.CAPABILITY_MISSING.value == "capability_missing"

    assert RoutingReasonCode.SELECTED.value == "selected"
