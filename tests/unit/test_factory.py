import pytest

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient
from agent_platform.llm.errors import LLMConfigurationError
from agent_platform.llm.factory import create_llm_client
from agent_platform.llm.openai_client import OpenAIClient


def test_factory_creates_openai_client() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
    )

    client = create_llm_client(settings)

    assert isinstance(client, OpenAIClient)
    assert isinstance(client, LLMClient)


def test_factory_is_case_insensitive() -> None:
    settings = Settings(
        llm_provider="OPENAI",
        openai_api_key="test-key",
    )

    client = create_llm_client(settings)

    assert isinstance(client, OpenAIClient)


def test_factory_rejects_unsupported_provider() -> None:
    settings = Settings(
        llm_provider="unsupported-provider",
        openai_api_key="test-key",
    )

    with pytest.raises(
        LLMConfigurationError,
        match="Unsupported LLM provider",
    ):
        create_llm_client(settings)


def test_factory_creates_client_for_routed_model() -> None:
    from agent_platform.llm.factory import create_llm_client_for_model
    from agent_platform.llm.model_definition import ModelDefinition
    from agent_platform.llm.workload import LLMWorkload

    settings = Settings(
        openai_api_key="test-key",
        llm_model="default-model",
    )

    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="routed-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )

    client = create_llm_client_for_model(
        settings,
        model,
    )

    assert isinstance(client, OpenAIClient)
    assert client.model == "routed-model"


def test_default_factory_still_uses_configured_model() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_model="default-model",
    )

    client = create_llm_client(settings)

    assert isinstance(client, OpenAIClient)
    assert client.model == "default-model"


def test_routed_factory_rejects_unsupported_provider() -> None:
    from agent_platform.llm.factory import create_llm_client_for_model
    from agent_platform.llm.model_definition import ModelDefinition
    from agent_platform.llm.workload import LLMWorkload

    settings = Settings(
        openai_api_key="test-key",
    )

    model = ModelDefinition(
        name="unsupported_model",
        provider="unsupported",
        provider_model="some-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )

    with pytest.raises(
        LLMConfigurationError,
        match="Unsupported LLM provider",
    ):
        create_llm_client_for_model(
            settings,
            model,
        )
