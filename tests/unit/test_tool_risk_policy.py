import pytest

from agent_platform.tools import (
    ToolApprovalContext,
    ToolApprovalRequiredError,
    ToolDefinition,
    ToolInput,
    ToolMetadata,
    ToolOutput,
    ToolRiskLevel,
    ToolRiskPolicy,
    ToolRiskPolicyNotConfiguredError,
    ToolRiskRule,
)


class PaymentInput(ToolInput):
    amount: int


class PaymentOutput(ToolOutput):
    transaction_id: str


def create_definition() -> ToolDefinition[PaymentInput, PaymentOutput]:
    return ToolDefinition(
        metadata=ToolMetadata(
            name="make_payment",
            description="Execute an approved customer payment.",
            version="1.0.0",
        ),
        input_type=PaymentInput,
        output_type=PaymentOutput,
    )


def test_risk_policy_returns_configured_risk_decision() -> None:
    policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="make_payment",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.HIGH,
                approval_required=True,
            ),
        )
    )

    decision = policy.evaluate(
        definition=create_definition(),
    )

    assert decision.risk_level is ToolRiskLevel.HIGH
    assert decision.approval_required is True


def test_risk_policy_fails_closed_when_rule_is_missing() -> None:
    policy = ToolRiskPolicy()

    with pytest.raises(
        ToolRiskPolicyNotConfiguredError,
        match="no risk policy configured",
    ):
        policy.evaluate(
            definition=create_definition(),
        )


def test_low_risk_tool_can_execute_without_approval() -> None:
    policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="make_payment",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.LOW,
                approval_required=False,
            ),
        )
    )

    definition = create_definition()

    decision = policy.evaluate(
        definition=definition,
    )

    policy.enforce_approval(
        definition=definition,
        decision=decision,
        approval_context=None,
    )


def test_high_risk_tool_is_denied_without_approval() -> None:
    policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="make_payment",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.HIGH,
                approval_required=True,
            ),
        )
    )

    definition = create_definition()

    decision = policy.evaluate(
        definition=definition,
    )

    with pytest.raises(
        ToolApprovalRequiredError,
        match="approval required",
    ):
        policy.enforce_approval(
            definition=definition,
            decision=decision,
            approval_context=None,
        )


def test_high_risk_tool_is_denied_when_approval_is_false() -> None:
    policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="make_payment",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.HIGH,
                approval_required=True,
            ),
        )
    )

    definition = create_definition()

    decision = policy.evaluate(
        definition=definition,
    )

    with pytest.raises(ToolApprovalRequiredError):
        policy.enforce_approval(
            definition=definition,
            decision=decision,
            approval_context=ToolApprovalContext(
                approved=False,
                approver_id="manager-123",
            ),
        )


def test_approved_execution_requires_approver_identity() -> None:
    policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="make_payment",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.HIGH,
                approval_required=True,
            ),
        )
    )

    definition = create_definition()

    decision = policy.evaluate(
        definition=definition,
    )

    with pytest.raises(
        ToolApprovalRequiredError,
        match="approver identity",
    ):
        policy.enforce_approval(
            definition=definition,
            decision=decision,
            approval_context=ToolApprovalContext(
                approved=True,
            ),
        )


def test_high_risk_tool_allows_valid_approval() -> None:
    policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="make_payment",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.HIGH,
                approval_required=True,
            ),
        )
    )

    definition = create_definition()

    decision = policy.evaluate(
        definition=definition,
    )

    policy.enforce_approval(
        definition=definition,
        decision=decision,
        approval_context=ToolApprovalContext(
            approved=True,
            approver_id="manager-123",
        ),
    )
