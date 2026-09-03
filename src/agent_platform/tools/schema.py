from typing import Any

from agent_platform.tools.contracts import ToolInput, ToolOutput
from agent_platform.tools.definition import ToolDefinition


class ToolSchemaBuilder:
    """Builds provider-neutral schemas for LLM-visible tools."""

    def build[
        InputT: ToolInput,
        OutputT: ToolOutput,
    ](
        self,
        *,
        definition: ToolDefinition[InputT, OutputT],
    ) -> dict[str, Any]:
        return {
            "name": definition.metadata.name,
            "version": definition.metadata.version,
            "description": definition.metadata.description,
            "input_schema": definition.input_type.model_json_schema(),
        }

    def build_many(
        self,
        *,
        definitions: tuple[
            ToolDefinition[ToolInput, ToolOutput],
            ...,
        ],
    ) -> tuple[dict[str, Any], ...]:
        return tuple(self.build(definition=definition) for definition in definitions)
