import pytest

from agent_platform.tools import (
    DuplicateToolBindingError,
    Tool,
    ToolBinding,
    ToolBindingNotFoundError,
    ToolBindingRegistry,
    ToolDefinition,
    ToolExecutionContext,
    ToolInput,
    ToolMetadata,
    ToolOutput,
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


def create_binding() -> ToolBinding[EchoInput, EchoOutput]:
    return ToolBinding(
        definition=ToolDefinition(
            metadata=ToolMetadata(
                name="echo",
                description="Echo a message.",
                version="1.0.0",
            ),
            input_type=EchoInput,
            output_type=EchoOutput,
        ),
        tool=EchoTool(),
    )


def test_binding_registry_registers_and_returns_binding() -> None:
    registry = ToolBindingRegistry()
    binding = create_binding()

    registry.register(binding)

    result = registry.get(
        name="echo",
        version="1.0.0",
    )

    assert result is binding


def test_binding_registry_rejects_duplicate_binding() -> None:
    registry = ToolBindingRegistry()
    binding = create_binding()

    registry.register(binding)

    with pytest.raises(DuplicateToolBindingError):
        registry.register(binding)


def test_binding_registry_rejects_missing_binding() -> None:
    registry = ToolBindingRegistry()

    with pytest.raises(ToolBindingNotFoundError):
        registry.get(
            name="missing",
            version="1.0.0",
        )
