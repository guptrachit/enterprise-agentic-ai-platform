from agent_platform.llm.errors import (
    LLMPromptActiveVersionNotSetError,
    LLMPromptAlreadyExistsError,
    LLMPromptNotFoundError,
)
from agent_platform.llm.prompt import PromptTemplate


class PromptRegistry:
    """In-memory registry of versioned prompt templates."""

    def __init__(self) -> None:
        self._prompts: dict[tuple[str, str], PromptTemplate] = {}
        self._active_versions: dict[str, str] = {}

    def register(self, prompt: PromptTemplate) -> None:
        """Register a prompt by name and version."""

        key = (prompt.name, prompt.version)

        if key in self._prompts:
            raise LLMPromptAlreadyExistsError(
                prompt.name,
                prompt.version,
            )

        self._prompts[key] = prompt

    def get(
        self,
        name: str,
        version: str,
    ) -> PromptTemplate:
        """Return an exact registered prompt version."""

        key = (name, version)

        try:
            return self._prompts[key]
        except KeyError as error:
            raise LLMPromptNotFoundError(
                name,
                version,
            ) from error

    def list_versions(
        self,
        name: str,
    ) -> tuple[str, ...]:
        """Return registered versions for a prompt name."""

        versions = sorted(
            version for prompt_name, version in self._prompts if prompt_name == name
        )

        return tuple(versions)

    def set_active(
        self,
        name: str,
        version: str,
    ) -> None:
        """Set the active version for a registered prompt."""

        self.get(
            name,
            version,
        )

        self._active_versions[name] = version

    def get_active(
        self,
        name: str,
    ) -> PromptTemplate:
        """Return the currently active prompt version."""

        try:
            version = self._active_versions[name]
        except KeyError as error:
            raise LLMPromptActiveVersionNotSetError(name) from error

        return self.get(
            name,
            version,
        )
