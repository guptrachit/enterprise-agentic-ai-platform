import pytest

from agent_platform.config import Settings
from agent_platform.llm.errors import LLMConfigurationError
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_configuration_validator import (
    validate_model_configuration,
)
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.workload import LLMWorkload


def test_model_configuration_validation_accepts_valid_configuration() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_primary",
                provider="openai",
                provider_model="primary-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
                capabilities=(ModelCapability.STRUCTURED_OUTPUT,),
            ),
            ModelConfig(
                name="classification_backup",
                provider="openai",
                provider_model="backup-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
                capabilities=(ModelCapability.STRUCTURED_OUTPUT,),
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

    validate_model_configuration(settings)


def test_model_configuration_validation_rejects_duplicate_names() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="duplicate_model",
                provider="openai",
                provider_model="model-a",
                workloads=(LLMWorkload.GENERAL,),
            ),
            ModelConfig(
                name="duplicate_model",
                provider="openai",
                provider_model="model-b",
                workloads=(LLMWorkload.GENERAL,),
            ),
        ),
    )

    with pytest.raises(
        LLMConfigurationError,
        match="Duplicate logical model names",
    ):
        validate_model_configuration(settings)


def test_model_configuration_validation_rejects_missing_policy_model() -> None:
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
        validate_model_configuration(settings)


def test_model_configuration_validation_rejects_unsupported_workload() -> None:
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
        validate_model_configuration(settings)


def test_model_configuration_validation_rejects_duplicate_capabilities() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="duplicate_capability_model",
                provider="openai",
                provider_model="duplicate-capability-model",
                workloads=(LLMWorkload.GENERAL,),
                capabilities=(
                    ModelCapability.TOOL_CALLING,
                    ModelCapability.TOOL_CALLING,
                ),
            ),
        ),
    )

    with pytest.raises(
        LLMConfigurationError,
        match="Duplicate capabilities",
    ):
        validate_model_configuration(settings)


def test_model_configuration_validation_accepts_legacy_model() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="legacy_model",
                provider="openai",
                provider_model="legacy-model",
                workloads=(LLMWorkload.GENERAL,),
            ),
        ),
    )

    validate_model_configuration(settings)


def test_model_configuration_validation_accepts_explicit_non_structured_model() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="tool_only_model",
                provider="openai",
                provider_model="tool-model",
                workloads=(LLMWorkload.GENERAL,),
                capabilities=(ModelCapability.TOOL_CALLING,),
            ),
        ),
    )

    validate_model_configuration(settings)
