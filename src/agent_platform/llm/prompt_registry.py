from agent_platform.llm.errors import (
    LLMPromptActiveDeprecationError,
    LLMPromptActiveVersionNotSetError,
    LLMPromptAlreadyExistsError,
    LLMPromptNotFoundError,
)
from agent_platform.llm.prompt import PromptTemplate
from agent_platform.llm.prompt_lifecycle import PromptStatus


class PromptRegistry:
    """In-memory registry of versioned prompt templates."""

    def __init__(self) -> None:
        self._prompts: dict[tuple[str, str], PromptTemplate] = {}
        self._active_versions: dict[str, str] = {}
        self._statuses: dict[tuple[str, str], PromptStatus] = {}

    def register(self, prompt: PromptTemplate) -> None:
        """Register a prompt version in draft state."""

        key = (prompt.name, prompt.version)

        if key in self._prompts:
            raise LLMPromptAlreadyExistsError(
                prompt.name,
                prompt.version,
            )

        self._prompts[key] = prompt
        self._statuses[key] = PromptStatus.DRAFT

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

    def get_status(
        self,
        name: str,
        version: str,
    ) -> PromptStatus:
        """Return the lifecycle status of a prompt version."""

        self.get(
            name,
            version,
        )

        return self._statuses[(name, version)]

    def set_active(
        self,
        name: str,
        version: str,
    ) -> None:
        """Promote a registered prompt version to active."""

        self.get(
            name,
            version,
        )

        previous_version = self._active_versions.get(name)

        if previous_version is not None and previous_version != version:
            previous_key = (
                name,
                previous_version,
            )

            self._statuses[previous_key] = PromptStatus.DEPRECATED

        key = (
            name,
            version,
        )

        self._statuses[key] = PromptStatus.ACTIVE
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

    def deprecate(
        self,
        name: str,
        version: str,
    ) -> None:
        """Mark a non-active prompt version as deprecated."""

        self.get(
            name,
            version,
        )

        active_version = self._active_versions.get(name)

        if active_version == version:
            raise LLMPromptActiveDeprecationError(
                name,
                version,
            )
        self._statuses[(name, version)] = PromptStatus.DEPRECATED
