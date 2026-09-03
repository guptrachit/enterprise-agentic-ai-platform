import pytest

from agent_platform.agents import (
    AgentDecision,
    AgentExecutionLimits,
    AgentExecutionService,
    AgentExecutionState,
    AgentFinalResponse,
    AgentGuardrailPolicy,
    AgentStepLimitExceededError,
    AgentToolLimitExceededError,
)
from agent_platform.tools import (
    Tool,
    ToolAuthorizationContext,
    ToolAuthorizationPolicy,
    ToolAuthorizationRule,
    ToolBinding,
    ToolBindingRegistry,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionService,
    ToolInput,
    ToolMetadata,
    ToolOutput,
    ToolRiskLevel,
    ToolRiskPolicy,
    ToolRiskRule,
    ToolSelection,
)


class EchoInput(ToolInput):
    message: str


class EchoOutput(ToolOutput):
    message: str


class EchoTool(Tool[EchoInput, EchoOutput]):
    @property
    def name(self) -> str:
        return "echo"

    async def execute(
        self,
        tool_input: EchoInput,
        context: ToolExecutionContext,
    ) -> EchoOutput:
        return EchoOutput(
            message=tool_input.message,
        )


class FinalOnlyModel:
    async def decide(
        self,
        *,
        state: AgentExecutionState,
    ) -> AgentDecision:
        return AgentDecision(
            final_response=AgentFinalResponse(
                content="No tool required.",
            )
        )


class TwoToolThenFinalModel:
    async def decide(
        self,
        *,
        state: AgentExecutionState,
    ) -> AgentDecision:
        if state.tool_execution_count < 2:
            return AgentDecision(
                tool_selection=ToolSelection(
                    tool_name="echo",
                    tool_version="1.0.0",
                    arguments={
                        "message": (f"tool-{state.tool_execution_count + 1}"),
                    },
                )
            )

        return AgentDecision(
            final_response=AgentFinalResponse(
                content="Completed two tools.",
            )
        )


class EndlessToolModel:
    async def decide(
        self,
        *,
        state: AgentExecutionState,
    ) -> AgentDecision:
        return AgentDecision(
            tool_selection=ToolSelection(
                tool_name="echo",
                tool_version="1.0.0",
                arguments={
                    "message": "again",
                },
            )
        )


def create_agent_service(
    model,
    *,
    limits: AgentExecutionLimits | None = None,
) -> AgentExecutionService:
    definition = ToolDefinition(
        metadata=ToolMetadata(
            name="echo",
            description="Echo a message.",
            version="1.0.0",
        ),
        input_type=EchoInput,
        output_type=EchoOutput,
    )

    binding = ToolBinding(
        definition=definition,
        tool=EchoTool(),
    )

    binding_registry = ToolBindingRegistry()
    binding_registry.register(binding)

    authorization_policy = ToolAuthorizationPolicy(
        rules=(
            ToolAuthorizationRule(
                tool_name="echo",
                tool_version="1.0.0",
                required_scopes=frozenset(
                    {
                        "tool:echo:execute",
                    }
                ),
            ),
        )
    )

    risk_policy = ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name="echo",
                tool_version="1.0.0",
                risk_level=ToolRiskLevel.LOW,
                approval_required=False,
            ),
        )
    )

    tool_execution_service = ToolExecutionService(
        authorization_policy=authorization_policy,
        risk_policy=risk_policy,
    )

    guardrail_policy = AgentGuardrailPolicy(
        limits=limits or AgentExecutionLimits(),
    )

    return AgentExecutionService(
        model=model,
        binding_registry=binding_registry,
        tool_execution_service=tool_execution_service,
        guardrail_policy=guardrail_policy,
    )


def create_execution_state() -> AgentExecutionState:
    return AgentExecutionState(
        user_message="Run agent",
        execution_context=ToolExecutionContext(
            correlation_id="corr-123",
        ),
        authorization_context=ToolAuthorizationContext(
            subject_id="user-123",
            granted_scopes=frozenset(
                {
                    "tool:echo:execute",
                }
            ),
        ),
    )


@pytest.mark.asyncio
async def test_agent_returns_final_response_without_tool() -> None:
    service = create_agent_service(
        FinalOnlyModel(),
    )

    result = await service.execute(
        state=create_execution_state(),
    )

    assert result.content == "No tool required."


@pytest.mark.asyncio
async def test_agent_executes_multiple_tools_before_final_response() -> None:
    service = create_agent_service(
        TwoToolThenFinalModel(),
    )

    result = await service.execute(
        state=create_execution_state(),
    )

    assert result.content == "Completed two tools."


@pytest.mark.asyncio
async def test_agent_stops_when_tool_limit_is_exceeded() -> None:
    service = create_agent_service(
        EndlessToolModel(),
        limits=AgentExecutionLimits(
            max_steps=10,
            max_tool_executions=2,
        ),
    )

    with pytest.raises(AgentToolLimitExceededError):
        await service.execute(
            state=create_execution_state(),
        )


@pytest.mark.asyncio
async def test_agent_stops_when_step_limit_is_exceeded() -> None:
    service = create_agent_service(
        EndlessToolModel(),
        limits=AgentExecutionLimits(
            max_steps=2,
            max_tool_executions=10,
        ),
    )

    with pytest.raises(AgentStepLimitExceededError):
        await service.execute(
            state=create_execution_state(),
        )
