from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient
from agent_platform.llm.errors import LLMConfigurationError
from agent_platform.llm.execution_service import LLMExecutionService
from agent_platform.llm.model_configuration_validator import (
    validate_model_configuration,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_policy_loader import load_model_policy
from agent_platform.llm.model_registry_loader import load_model_registry
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


def create_model_router(
    settings: Settings,
    policy: ModelPolicy | None = None,
) -> ModelRouter:
    """Create a validated model router from configuration."""

    if policy is None:
        validate_model_configuration(settings)

    registry = load_model_registry(settings)

    resolved_policy = policy if policy is not None else load_model_policy(settings)

    return ModelRouter(
        registry=registry,
        policy=resolved_policy,
    )


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


def create_configured_llm_execution_service(
    settings: Settings,
    policy: ModelPolicy | None = None,
) -> LLMExecutionService:
    """Create a fully configuration-driven routed execution service."""

    router = create_model_router(
        settings,
        policy,
    )

    return create_llm_execution_service(
        settings,
        router,
    )
