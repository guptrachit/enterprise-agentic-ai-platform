from typing import Any, Protocol

from agent_platform.tools.registry import (
    ToolNotFoundError,
    ToolRegistry,
)
from agent_platform.tools.schema import ToolSchemaBuilder
from agent_platform.tools.selection import (
    ToolSelection,
    ToolSelectionError,
    ToolSelectionRequest,
)


class ToolSelectionModel(Protocol):
    """Provider-neutral model capable of selecting governed tools."""

    async def select_tool(
        self,
        *,
        user_message: str,
        tools: tuple[dict[str, Any], ...],
    ) -> ToolSelection | None:
        """Select a tool from the supplied governed tool catalog."""


class ToolSelectionService:
    """Coordinates governed LLM tool selection."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        model: ToolSelectionModel,
        schema_builder: ToolSchemaBuilder | None = None,
    ) -> None:
        self._registry = registry
        self._model = model
        self._schema_builder = schema_builder or ToolSchemaBuilder()

    async def select(
        self,
        *,
        request: ToolSelectionRequest,
    ) -> ToolSelection | None:
        definitions = self._registry.list_definitions()

        schemas = self._schema_builder.build_many(
            definitions=definitions,
        )

        selection = await self._model.select_tool(
            user_message=request.user_message,
            tools=schemas,
        )

        if selection is None:
            return None

        try:
            self._registry.get(
                name=selection.tool_name,
                version=selection.tool_version,
            )
        except ToolNotFoundError as exc:
            raise ToolSelectionError(
                f"LLM selected unknown tool "
                f"{selection.tool_name}:{selection.tool_version}"
            ) from exc

        return selection
