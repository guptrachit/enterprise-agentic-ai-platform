from agent_platform.tools.contracts import ToolInput, ToolOutput
from agent_platform.tools.metadata import ToolMetadata


class ToolDefinition[
    InputT: ToolInput,
    OutputT: ToolOutput,
]:
    """Complete platform definition of a governed tool."""

    def __init__(
        self,
        *,
        metadata: ToolMetadata,
        input_type: type[InputT],
        output_type: type[OutputT],
    ) -> None:
        if not issubclass(input_type, ToolInput):
            raise TypeError("input_type must inherit from ToolInput")

        if not issubclass(output_type, ToolOutput):
            raise TypeError("output_type must inherit from ToolOutput")

        self._metadata = metadata
        self._input_type = input_type
        self._output_type = output_type

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def input_type(self) -> type[InputT]:
        return self._input_type

    @property
    def output_type(self) -> type[OutputT]:
        return self._output_type
