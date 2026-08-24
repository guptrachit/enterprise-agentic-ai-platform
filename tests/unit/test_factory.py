import pytest

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient
from agent_platform.llm.errors import LLMConfigurationError
from agent_platform.llm.factory import (
    create_configured_llm_execution_service,
    create_llm_client,
    create_llm_client_for_model,
    create_model_router,
)
from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.openai_client import OpenAIClient
from agent_platform.llm.workload import LLMWorkload


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


def test_create_model_router_uses_configured_registry() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="fast_general",
                provider="openai",
                provider_model="gpt-5-mini",
                workloads=(
                    LLMWorkload.GENERAL,
                    LLMWorkload.CLASSIFICATION,
                ),
            ),
        ),
    )

    policy = ModelPolicy(
        assignments={
            LLMWorkload.CLASSIFICATION: "fast_general",
        }
    )

    router = create_model_router(
        settings,
        policy,
    )

    model = router.route(LLMWorkload.CLASSIFICATION)

    assert model.name == "fast_general"
    assert model.provider == "openai"
    assert model.provider_model == "gpt-5-mini"


def test_create_configured_execution_service_uses_settings_models() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_model",
                provider="openai",
                provider_model="configured-classification-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
            ),
        ),
    )

    policy = ModelPolicy(
        assignments={
            LLMWorkload.CLASSIFICATION: "classification_model",
        }
    )

    service = create_configured_llm_execution_service(
        settings,
        policy,
    )

    model = service.router.route(LLMWorkload.CLASSIFICATION)

    assert model.name == "classification_model"
    assert model.provider_model == "configured-classification-model"


def test_create_model_router_uses_configured_policy() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_model",
                provider="openai",
                provider_model="configured-classification-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
            ),
        ),
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.CLASSIFICATION: "classification_model",
            }
        ),
    )

    router = create_model_router(settings)

    model = router.route(LLMWorkload.CLASSIFICATION)

    assert model.name == "classification_model"
    assert model.provider_model == "configured-classification-model"


def test_configured_execution_service_uses_configured_policy() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_primary",
                provider="openai",
                provider_model="primary-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
            ),
            ModelConfig(
                name="classification_backup",
                provider="openai",
                provider_model="backup-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
            ),
        ),
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.CLASSIFICATION: (
                    "classification_primary",
                    "classification_backup",
                ),
            }
        ),
    )

    service = create_configured_llm_execution_service(settings)

    candidates = service.router.route_candidates(LLMWorkload.CLASSIFICATION)

    assert tuple(model.name for model in candidates) == (
        "classification_primary",
        "classification_backup",
    )


def test_configured_router_rejects_missing_policy_model() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_primary",
                provider="openai",
                provider_model="primary-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
            ),
        ),
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.CLASSIFICATION: (
                    "classification_primary",
                    "missing_backup",
                ),
            }
        ),
    )

    with pytest.raises(
        LLMConfigurationError,
        match="missing_backup",
    ):
        create_model_router(settings)


def test_configured_router_rejects_workload_mismatch() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="general_model",
                provider="openai",
                provider_model="general-model",
                workloads=(LLMWorkload.GENERAL,),
            ),
        ),
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.REASONING: "general_model",
            }
        ),
    )

    with pytest.raises(
        LLMConfigurationError,
        match="does not support workload 'reasoning'",
    ):
        create_model_router(settings)
