from agent_platform.tools.contracts import ToolInput, ToolOutput
from agent_platform.tools.definition import ToolDefinition


class ToolRegistryError(Exception):
    """Base error raised by the tool registry."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a tool with the same identity is registered twice."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


class ToolRegistry:
    """Authoritative catalog of tool definitions known to the platform."""

    def __init__(self) -> None:
        self._definitions: dict[
            tuple[str, str],
            ToolDefinition[ToolInput, ToolOutput],
        ] = {}

    def register(
        self,
        definition: ToolDefinition[ToolInput, ToolOutput],
    ) -> None:
        key = (
            definition.metadata.name,
            definition.metadata.version,
        )

        if key in self._definitions:
            raise DuplicateToolError(
                "tool already registered: "
                f"{definition.metadata.name}:{definition.metadata.version}"
            )

        self._definitions[key] = definition

    def get(
        self,
        *,
        name: str,
        version: str,
    ) -> ToolDefinition[ToolInput, ToolOutput]:
        key = (name, version)

        try:
            return self._definitions[key]
        except KeyError as exc:
            raise ToolNotFoundError(f"tool not registered: {name}:{version}") from exc

    def contains(
        self,
        *,
        name: str,
        version: str,
    ) -> bool:
        return (name, version) in self._definitions

    def list_definitions(
        self,
    ) -> tuple[ToolDefinition[ToolInput, ToolOutput], ...]:
        return tuple(self._definitions.values())
