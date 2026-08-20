from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient
from agent_platform.llm.errors import LLMConfigurationError
from agent_platform.llm.openai_client import OpenAIClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Create an LLM client based on configured provider."""

    provider = settings.llm_provider.strip().lower()

    if provider == "openai":
        return OpenAIClient(settings)

    raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")
