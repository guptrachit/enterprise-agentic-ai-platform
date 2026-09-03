from agent_platform.tools.binding import ToolBinding
from agent_platform.tools.contracts import ToolInput, ToolOutput


class ToolBindingRegistryError(Exception):
    """Base error raised by the executable tool binding registry."""


class DuplicateToolBindingError(ToolBindingRegistryError):
    """Raised when an executable binding is registered twice."""


class ToolBindingNotFoundError(ToolBindingRegistryError):
    """Raised when an executable tool binding cannot be found."""


class ToolBindingRegistry:
    """Registry of governed executable tool bindings."""

    def __init__(self) -> None:
        self._bindings: dict[
            tuple[str, str],
            ToolBinding[ToolInput, ToolOutput],
        ] = {}

    def register(
        self,
        binding: ToolBinding[ToolInput, ToolOutput],
    ) -> None:
        key = (
            binding.definition.metadata.name,
            binding.definition.metadata.version,
        )

        if key in self._bindings:
            raise DuplicateToolBindingError(
                f"tool binding already registered: {key[0]}:{key[1]}"
            )

        self._bindings[key] = binding

    def get(
        self,
        *,
        name: str,
        version: str,
    ) -> ToolBinding[ToolInput, ToolOutput]:
        key = (name, version)

        try:
            return self._bindings[key]
        except KeyError as exc:
            raise ToolBindingNotFoundError(
                f"tool binding not registered: {name}:{version}"
            ) from exc
