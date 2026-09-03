from typing import Any

from pydantic import ValidationError

from agent_platform.tools.contracts import ToolInput, ToolOutput
from agent_platform.tools.definition import ToolDefinition


class ToolValidationError(Exception):
    """Base error raised when tool contract validation fails."""


class ToolInputValidationError(ToolValidationError):
    """Raised when proposed tool input violates the input contract."""


class ToolOutputValidationError(ToolValidationError):
    """Raised when tool output violates the output contract."""


class ToolValidator:
    """Validates tool inputs and outputs against registered contracts."""

    def validate_input[
        InputT: ToolInput,
        OutputT: ToolOutput,
    ](
        self,
        *,
        definition: ToolDefinition[InputT, OutputT],
        payload: Any,
    ) -> InputT:
        try:
            return definition.input_type.model_validate(payload)
        except ValidationError as exc:
            raise ToolInputValidationError(
                f"invalid input for tool "
                f"{definition.metadata.name}:{definition.metadata.version}"
            ) from exc

    def validate_output[
        InputT: ToolInput,
        OutputT: ToolOutput,
    ](
        self,
        *,
        definition: ToolDefinition[InputT, OutputT],
        payload: Any,
    ) -> OutputT:
        try:
            return definition.output_type.model_validate(payload)
        except ValidationError as exc:
            raise ToolOutputValidationError(
                f"invalid output for tool "
                f"{definition.metadata.name}:{definition.metadata.version}"
            ) from exc
