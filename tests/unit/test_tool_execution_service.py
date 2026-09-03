import pytest

from agent_platform.tools import (
    InMemoryToolTelemetrySink,
    Tool,
    ToolApprovalContext,
    ToolApprovalRequiredError,
    ToolAuthorizationContext,
    ToolAuthorizationDeniedError,
    ToolAuthorizationPolicy,
    ToolAuthorizationRule,
    ToolBinding,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionService,
    ToolExecutionStatus,
    ToolInput,
    ToolInputValidationError,
    ToolMetadata,
    ToolOutput,
    ToolOutputValidationError,
    ToolRetryableError,
    ToolRetryPolicy,
    ToolRiskLevel,
    ToolRiskPolicy,
    ToolRiskRule,
)


def create_risk_policy(
    *,
    tool_name: str,
    approval_required: bool = False,
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
) -> ToolRiskPolicy:
    return ToolRiskPolicy(
        rules=(
            ToolRiskRule(
                tool_name=tool_name,
                tool_version="1.0.0",
                risk_level=risk_level,
                approval_required=approval_required,
            ),
        )
    )


class EchoInput(ToolInput):
    message: str


class EchoOutput(ToolOutput):
    message: str
    correlation_id: str


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
            correlation_id=context.correlation_id,
        )


class InvalidOutputTool(Tool[EchoInput, EchoOutput]):
    @property
    def name(self) -> str:
        return "invalid_output"

    async def execute(
        self,
        tool_input: EchoInput,
        context: ToolExecutionContext,
    ) -> EchoOutput:
        return {
            "message": tool_input.message,
            "unexpected": True,
        }


def create_binding(
    tool: Tool[EchoInput, EchoOutput],
) -> ToolBinding[EchoInput, EchoOutput]:
    return ToolBinding(
        definition=ToolDefinition(
            metadata=ToolMetadata(
                name=tool.name,
                description="Echo a message for testing.",
                version="1.0.0",
            ),
            input_type=EchoInput,
            output_type=EchoOutput,
        ),
        tool=tool,
    )


def create_authorization_policy(
    *,
    tool_name: str,
) -> ToolAuthorizationPolicy:
    return ToolAuthorizationPolicy(
        rules=(
            ToolAuthorizationRule(
                tool_name=tool_name,
                tool_version="1.0.0",
                required_scopes=frozenset(
                    {
                        f"tool:{tool_name}:execute",
                    }
                ),
            ),
        )
    )


def create_authorization_context(
    *,
    tool_name: str,
) -> ToolAuthorizationContext:
    return ToolAuthorizationContext(
        subject_id="user-123",
        granted_scopes=frozenset(
            {
                f"tool:{tool_name}:execute",
            }
        ),
    )


@pytest.mark.asyncio
async def test_execution_service_executes_authorized_tool() -> None:
    binding = create_binding(EchoTool())

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="echo",
        ),
        risk_policy=create_risk_policy(
            tool_name="echo",
        ),
    )

    result = await service.execute(
        binding=binding,
        payload={
            "message": "hello",
        },
        context=ToolExecutionContext(
            correlation_id="corr-123",
        ),
        authorization_context=create_authorization_context(
            tool_name="echo",
        ),
    )

    assert result.message == "hello"
    assert result.correlation_id == "corr-123"


@pytest.mark.asyncio
async def test_execution_service_rejects_invalid_input_before_execution() -> None:
    binding = create_binding(EchoTool())

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="echo",
        ),
        risk_policy=create_risk_policy(
            tool_name="invalid_output",
        ),
    )

    with pytest.raises(ToolInputValidationError):
        await service.execute(
            binding=binding,
            payload={
                "unexpected": "not-allowed",
            },
            context=ToolExecutionContext(
                correlation_id="corr-123",
            ),
            authorization_context=create_authorization_context(
                tool_name="echo",
            ),
        )


@pytest.mark.asyncio
async def test_execution_service_denies_unauthorized_tool() -> None:
    binding = create_binding(EchoTool())

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="echo",
        ),
        risk_policy=create_risk_policy(
            tool_name="invalid_output",
        ),
    )

    with pytest.raises(ToolAuthorizationDeniedError):
        await service.execute(
            binding=binding,
            payload={
                "message": "hello",
            },
            context=ToolExecutionContext(
                correlation_id="corr-123",
            ),
            authorization_context=ToolAuthorizationContext(
                subject_id="user-123",
                granted_scopes=frozenset(),
            ),
        )


@pytest.mark.asyncio
async def test_execution_service_rejects_invalid_output() -> None:
    binding = create_binding(InvalidOutputTool())

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="invalid_output",
        ),
        risk_policy=create_risk_policy(
            tool_name="invalid_output",
        ),
    )

    with pytest.raises(ToolOutputValidationError):
        await service.execute(
            binding=binding,
            payload={
                "message": "hello",
            },
            context=ToolExecutionContext(
                correlation_id="corr-123",
            ),
            authorization_context=create_authorization_context(
                tool_name="invalid_output",
            ),
        )


@pytest.mark.asyncio
async def test_execution_service_requires_approval_before_tool_execution() -> None:
    binding = create_binding(EchoTool())

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="echo",
        ),
        risk_policy=create_risk_policy(
            tool_name="echo",
            risk_level=ToolRiskLevel.HIGH,
            approval_required=True,
        ),
    )

    with pytest.raises(ToolApprovalRequiredError):
        await service.execute(
            binding=binding,
            payload={
                "message": "hello",
            },
            context=ToolExecutionContext(
                correlation_id="corr-123",
            ),
            authorization_context=create_authorization_context(
                tool_name="echo",
            ),
        )


@pytest.mark.asyncio
async def test_execution_service_executes_after_required_approval() -> None:
    binding = create_binding(EchoTool())

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="echo",
        ),
        risk_policy=create_risk_policy(
            tool_name="echo",
            risk_level=ToolRiskLevel.HIGH,
            approval_required=True,
        ),
    )

    result = await service.execute(
        binding=binding,
        payload={
            "message": "hello",
        },
        context=ToolExecutionContext(
            correlation_id="corr-123",
        ),
        authorization_context=create_authorization_context(
            tool_name="echo",
        ),
        approval_context=ToolApprovalContext(
            approved=True,
            approver_id="manager-123",
        ),
    )

    assert result.message == "hello"


class RetryableEchoTool(Tool[EchoInput, EchoOutput]):
    def __init__(self) -> None:
        self.attempts = 0

    @property
    def name(self) -> str:
        return "retryable_echo"

    async def execute(
        self,
        tool_input: EchoInput,
        context: ToolExecutionContext,
    ) -> EchoOutput:
        self.attempts += 1

        if self.attempts == 1:
            raise ToolRetryableError("temporary downstream failure")

        return EchoOutput(
            message=tool_input.message,
            correlation_id=context.correlation_id,
        )


@pytest.mark.asyncio
async def test_execution_service_retries_retryable_tool_failure() -> None:
    tool = RetryableEchoTool()
    binding = create_binding(tool)

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="retryable_echo",
        ),
        risk_policy=create_risk_policy(
            tool_name="retryable_echo",
        ),
        retry_policy=ToolRetryPolicy(
            max_attempts=2,
            timeout_seconds=1,
        ),
    )

    result = await service.execute(
        binding=binding,
        payload={
            "message": "hello",
        },
        context=ToolExecutionContext(
            correlation_id="corr-123",
        ),
        authorization_context=create_authorization_context(
            tool_name="retryable_echo",
        ),
    )

    assert result.message == "hello"
    assert tool.attempts == 2


@pytest.mark.asyncio
async def test_execution_service_emits_success_telemetry() -> None:
    binding = create_binding(EchoTool())
    telemetry_sink = InMemoryToolTelemetrySink()

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="echo",
        ),
        risk_policy=create_risk_policy(
            tool_name="echo",
        ),
        telemetry_sink=telemetry_sink,
    )

    await service.execute(
        binding=binding,
        payload={
            "message": "hello",
        },
        context=ToolExecutionContext(
            correlation_id="corr-telemetry-success",
        ),
        authorization_context=create_authorization_context(
            tool_name="echo",
        ),
    )

    assert len(telemetry_sink.events) == 1

    event = telemetry_sink.events[0]

    assert event.tool_name == "echo"
    assert event.tool_version == "1.0.0"
    assert event.subject_id == "user-123"
    assert event.correlation_id == "corr-telemetry-success"
    assert event.status == ToolExecutionStatus.SUCCESS
    assert event.error_type is None
    assert event.duration_ms >= 0


@pytest.mark.asyncio
async def test_execution_service_emits_failure_telemetry() -> None:
    binding = create_binding(InvalidOutputTool())
    telemetry_sink = InMemoryToolTelemetrySink()

    service = ToolExecutionService(
        authorization_policy=create_authorization_policy(
            tool_name="invalid_output",
        ),
        risk_policy=create_risk_policy(
            tool_name="invalid_output",
        ),
        telemetry_sink=telemetry_sink,
    )

    with pytest.raises(ToolOutputValidationError):
        await service.execute(
            binding=binding,
            payload={
                "message": "hello",
            },
            context=ToolExecutionContext(
                correlation_id="corr-telemetry-failure",
            ),
            authorization_context=create_authorization_context(
                tool_name="invalid_output",
            ),
        )

    assert len(telemetry_sink.events) == 1

    event = telemetry_sink.events[0]

    assert event.tool_name == "invalid_output"
    assert event.status == ToolExecutionStatus.FAILURE
    assert event.error_type == "ToolOutputValidationError"
    assert event.correlation_id == "corr-telemetry-failure"
