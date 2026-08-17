from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface for interacting with an LLM provider."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError
