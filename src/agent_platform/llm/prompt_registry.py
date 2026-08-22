from agent_platform.llm.errors import (
    LLMPromptAlreadyExistsError,
    LLMPromptNotFoundError,
)
from agent_platform.llm.prompt import PromptTemplate


class PromptRegistry:
    """In-memory registry of versioned prompt templates."""

    def __init__(self) -> None:
        self._prompts: dict[tuple[str, str], PromptTemplate] = {}

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
