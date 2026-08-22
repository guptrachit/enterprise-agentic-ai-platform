from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient
from agent_platform.llm.errors import LLMConfigurationError
from agent_platform.llm.execution_service import LLMExecutionService
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.openai_client import OpenAIClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Create the default LLM client from application configuration."""

    provider = settings.llm_provider.strip().lower()

    if provider == "openai":
        return OpenAIClient(settings)

    raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")


def create_llm_client_for_model(
    settings: Settings,
    model: ModelDefinition,
) -> LLMClient:
    """Create an LLM client for a routed model definition."""

    provider = model.provider.strip().lower()

    if provider == "openai":
        return OpenAIClient(
            settings,
            model=model.provider_model,
        )

    raise LLMConfigurationError(f"Unsupported LLM provider: {model.provider}")


def create_llm_execution_service(
    settings: Settings,
    router: ModelRouter,
) -> LLMExecutionService:
    """Create an execution service backed by routed provider clients."""

    return LLMExecutionService(
        router=router,
        client_factory=lambda model: create_llm_client_for_model(
            settings,
            model,
        ),
    )
