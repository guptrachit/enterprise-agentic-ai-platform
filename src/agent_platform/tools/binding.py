from agent_platform.tools.base import Tool
from agent_platform.tools.contracts import ToolInput, ToolOutput
from agent_platform.tools.definition import ToolDefinition


class ToolBinding[
    InputT: ToolInput,
    OutputT: ToolOutput,
]:
    """Binds a governed tool definition to its executable implementation."""

    def __init__(
        self,
        *,
        definition: ToolDefinition[InputT, OutputT],
        tool: Tool[InputT, OutputT],
    ) -> None:
        self._definition = definition
        self._tool = tool

    @property
    def definition(self) -> ToolDefinition[InputT, OutputT]:
        return self._definition

    @property
    def tool(self) -> Tool[InputT, OutputT]:
        return self._tool
