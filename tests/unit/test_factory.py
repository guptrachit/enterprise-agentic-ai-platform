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
