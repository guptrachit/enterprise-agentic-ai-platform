from abc import ABC, abstractmethod

from agent_platform.tools.contracts import (
    ToolExecutionContext,
    ToolInput,
    ToolOutput,
)


class Tool[InputT: ToolInput, OutputT: ToolOutput](ABC):
    """Provider-neutral contract implemented by every platform tool."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable logical name of the tool."""

    @abstractmethod
    async def execute(
        self,
        tool_input: InputT,
        context: ToolExecutionContext,
    ) -> OutputT:
        """Execute the tool using validated input and platform context."""
