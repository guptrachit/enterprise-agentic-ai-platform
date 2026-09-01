import pytest

from agent_platform.tools import (
    Tool,
    ToolExecutionContext,
    ToolInput,
    ToolOutput,
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


@pytest.mark.asyncio
async def test_tool_contract_executes_with_typed_input_and_output() -> None:
    tool = EchoTool()

    tool_input = EchoInput(message="hello")

    context = ToolExecutionContext(
        correlation_id="corr-123",
    )

    result = await tool.execute(
        tool_input=tool_input,
        context=context,
    )

    assert tool.name == "echo"
    assert result.message == "hello"
    assert result.correlation_id == "corr-123"


def test_tool_input_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        EchoInput(
            message="hello",
            unexpected_field="not-allowed",
        )
