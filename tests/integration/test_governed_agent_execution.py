import pytest

from agent_platform.agents import (
    AgentApprovalRequired,
    AgentDecision,
    AgentExecutionLimits,
    AgentExecutionService,
    AgentExecutionState,
    AgentFinalResponse,
    AgentGuardrailPolicy,
)
from agent_platform.tools import (
    InMemoryToolTelemetrySink,
    Tool,
    ToolAuthorizationContext,
    ToolAuthorizationPolicy,
    ToolAuthorizationRule,
    ToolBinding,
    ToolBindingRegistry,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionService,
    ToolExecutionStatus,
    ToolInput,
    ToolMetadata,
    ToolOutput,
    ToolRetryableError,
    ToolRetryPolicy,
    ToolRiskLevel,
    ToolRiskPolicy,
    ToolRiskRule,
    ToolSelection,
)


class PaymentInput(ToolInput):
    customer_id: str
    amount: float


class PaymentOutput(ToolOutput):
    customer_id: str
    amount: float
    status: str


class RetryOncePaymentTool(Tool[PaymentInput, PaymentOutput]):
    """Simulates a transient downstream failure before success."""

    def __init__(self) -> None:
        self.attempts = 0

    @property
    def name(self) -> str:
        return "issue_payment"

    async def execute(
        self,
        tool_input: PaymentInput,
        context: ToolExecutionContext,
    ) -> PaymentOutput:
        self.attempts += 1

        if self.attempts == 1:
            raise ToolRetryableError("temporary payment service failure")

        return PaymentOutput(
            customer_id=tool_input.customer_id,
            amount=tool_input.amount,
            status="completed",
        )


class PaymentAgentModel:
    """Proposes one high-risk action and then returns a final answer."""

    async def decide(
        self,
        *,
        state: AgentExecutionState,
    ) -> AgentDecision:
        if not state.observations:
            return AgentDecision(
                tool_selection=ToolSelection(
                    tool_name="issue_payment",
                    tool_version="1.0.0",
                    arguments={
                        "customer_id": "customer-123",
                        "amount": 125.50,
                    },
                )
            )

        observation = state.observations[-1]

        return AgentDecision(
            final_response=AgentFinalResponse(
                content=(
                    "Payment completed for "
                    f"{observation.output['customer_id']} "
                    f"for ${observation.output['amount']}."
                ),
            )
        )


def create_agent_runtime():
    tool = RetryOncePaymentTool()

    definition = ToolDefinition(
        metadata=ToolMetadata(
            name="issue_payment",
            version="1.0.0",
            description=("Issue a governed customer payment."),
        ),
        input_type=PaymentInput,
        output_type=PaymentOutput,
    )

    binding = ToolBinding(
        definition=definition,
        tool=tool,
    )

    binding_registry = ToolBindingRegistry()
    binding_registry.register(binding)

    authorization_policy = ToolAuthorizationPolicy(
        rules=(
            ToolAuthorizationRule(
                tool_name="issue_payment",
                tool_version="1.0.0",
                required_scopes=frozenset(
                    {
                        "tool:issue_payment:execute",
                    }
                ),
            ),
        )
    )

    risk_policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="issue_payment",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.HIGH,
                approval_required=True,
            ),
        )
    )

    telemetry_sink = InMemoryToolTelemetrySink()

    tool_execution_service = ToolExecutionService(
        authorization_policy=authorization_policy,
        risk_policy=risk_policy,
        retry_policy=ToolRetryPolicy(
            max_attempts=2,
            timeout_seconds=5.0,
        ),
        telemetry_sink=telemetry_sink,
    )

    agent_service = AgentExecutionService(
        model=PaymentAgentModel(),
        binding_registry=binding_registry,
        tool_execution_service=tool_execution_service,
        guardrail_policy=AgentGuardrailPolicy(
            limits=AgentExecutionLimits(
                max_steps=5,
                max_tool_executions=2,
            )
        ),
    )

    return (
        agent_service,
        tool,
        telemetry_sink,
    )


def create_execution_state() -> AgentExecutionState:
    return AgentExecutionState(
        user_message=("Issue the approved payment for customer 123."),
        execution_context=ToolExecutionContext(
            correlation_id="corr-payment-123",
        ),
        authorization_context=ToolAuthorizationContext(
            subject_id="agent-user-123",
            granted_scopes=frozenset(
                {
                    "tool:issue_payment:execute",
                }
            ),
        ),
    )


@pytest.mark.asyncio
async def test_governed_agent_pauses_for_human_approval() -> None:
    service, tool, telemetry = create_agent_runtime()

    result = await service.execute(
        state=create_execution_state(),
    )

    assert isinstance(
        result,
        AgentApprovalRequired,
    )

    assert result.tool_selection.tool_name == "issue_payment"

    assert result.tool_selection.arguments == {
        "customer_id": "customer-123",
        "amount": 125.50,
    }

    assert tool.attempts == 0
    assert telemetry.events == ()


@pytest.mark.asyncio
async def test_governed_agent_executes_exact_action_after_approval() -> None:
    service, tool, telemetry = create_agent_runtime()

    paused = await service.execute(
        state=create_execution_state(),
    )

    assert isinstance(
        paused,
        AgentApprovalRequired,
    )

    service.approval_store.approve(
        approval_id=paused.approval_id,
        approver_id="manager-456",
    )

    result = await service.resume(
        approval_id=paused.approval_id,
    )

    assert isinstance(
        result,
        AgentFinalResponse,
    )

    assert result.content == ("Payment completed for customer-123 for $125.5.")

    # First invocation fails transiently.
    # Second invocation succeeds.
    assert tool.attempts == 2

    assert len(telemetry.events) == 1

    event = telemetry.events[0]

    assert event.tool_name == "issue_payment"
    assert event.tool_version == "1.0.0"

    assert event.subject_id == "agent-user-123"

    assert event.correlation_id == "corr-payment-123"

    assert event.risk_level is ToolRiskLevel.HIGH

    assert event.approval_required is True

    assert event.approver_id == "manager-456"

    assert event.status is ToolExecutionStatus.SUCCESS
    assert event.attempt_count == 2

    assert event.error_type is None


@pytest.mark.asyncio
async def test_human_approval_does_not_override_authorization() -> None:
    service, tool, _ = create_agent_runtime()

    unauthorized_state = AgentExecutionState(
        user_message="Issue payment.",
        execution_context=ToolExecutionContext(
            correlation_id="corr-unauthorized",
        ),
        authorization_context=ToolAuthorizationContext(
            subject_id="unauthorized-user",
            granted_scopes=frozenset(),
        ),
    )

    paused = await service.execute(
        state=unauthorized_state,
    )

    assert isinstance(
        paused,
        AgentApprovalRequired,
    )

    service.approval_store.approve(
        approval_id=paused.approval_id,
        approver_id="manager-456",
    )

    from agent_platform.tools import (
        ToolAuthorizationDeniedError,
    )

    with pytest.raises(
        ToolAuthorizationDeniedError,
    ):
        await service.resume(
            approval_id=paused.approval_id,
        )

    assert tool.attempts == 0
