from unittest.mock import AsyncMock

import pytest

from agent_platform.config import Settings
from agent_platform.llm.errors import (
    LLMConfigurationError,
    LLMTransientError,
)
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.factory import (
    create_configured_llm_execution_service,
)
from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.workload import LLMWorkload


@pytest.mark.asyncio
async def test_configuration_bootstraps_complete_routed_execution(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_primary",
                provider="openai",
                provider_model="primary-provider-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
            ),
            ModelConfig(
                name="classification_backup",
                provider="openai",
                provider_model="backup-provider-model",
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

    assert tuple(model.provider_model for model in candidates) == (
        "primary-provider-model",
        "backup-provider-model",
    )

    primary_client = AsyncMock()

    expected_response = object()
    primary_client.generate.return_value = expected_response

    def fake_client_factory(model):
        captured["logical_model"] = model.name
        captured["provider"] = model.provider
        captured["provider_model"] = model.provider_model

        return primary_client

    monkeypatch.setattr(
        service,
        "client_factory",
        fake_client_factory,
    )

    request = LLMExecutionRequest(
        prompt="Classify this support ticket.",
        workload=LLMWorkload.CLASSIFICATION,
        correlation_id="corr-bootstrap-001",
        prompt_name="ticket_classifier",
        prompt_version="1.0",
    )

    result = await service.execute(request)

    assert result is expected_response

    assert captured == {
        "logical_model": "classification_primary",
        "provider": "openai",
        "provider_model": "primary-provider-model",
    }

    primary_client.generate.assert_awaited_once_with(
        "Classify this support ticket.",
        correlation_id="corr-bootstrap-001",
        prompt_name="ticket_classifier",
        prompt_version="1.0",
        workload="classification",
        logical_model="classification_primary",
        fallback_used=False,
        fallback_from=None,
        fallback_reason=None,
        allowed_providers=None,
        max_cost_tier=None,
        max_latency_tier=None,
        prefer_lower_cost=False,
        prefer_lower_latency=False,
        preferred_providers=None,
        preferred_cost_tier=None,
        preferred_latency_tier=None,
    )


@pytest.mark.asyncio
async def test_configuration_bootstrap_preserves_fallback_order(
    monkeypatch,
) -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_primary",
                provider="openai",
                provider_model="primary-provider-model",
                workloads=(LLMWorkload.CLASSIFICATION,),
            ),
            ModelConfig(
                name="classification_backup",
                provider="openai",
                provider_model="backup-provider-model",
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

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMTransientError()

    backup_client = AsyncMock()

    expected_response = object()
    backup_client.generate.return_value = expected_response

    def fake_client_factory(model):
        if model.name == "classification_primary":
            return primary_client

        return backup_client

    monkeypatch.setattr(
        service,
        "client_factory",
        fake_client_factory,
    )

    result = await service.execute(
        LLMExecutionRequest(
            prompt="Classify this support ticket.",
            workload=LLMWorkload.CLASSIFICATION,
        )
    )

    assert result is expected_response

    primary_client.generate.assert_awaited_once_with(
        "Classify this support ticket.",
        correlation_id=None,
        prompt_name=None,
        prompt_version=None,
        workload="classification",
        logical_model="classification_primary",
        fallback_used=False,
        fallback_from=None,
        fallback_reason=None,
        allowed_providers=None,
        max_cost_tier=None,
        max_latency_tier=None,
        prefer_lower_cost=False,
        prefer_lower_latency=False,
        preferred_providers=None,
        preferred_cost_tier=None,
        preferred_latency_tier=None,
    )

    backup_client.generate.assert_awaited_once_with(
        "Classify this support ticket.",
        correlation_id=None,
        prompt_name=None,
        prompt_version=None,
        workload="classification",
        logical_model="classification_backup",
        fallback_used=True,
        fallback_from="classification_primary",
        fallback_reason="LLMTransientError",
        allowed_providers=None,
        max_cost_tier=None,
        max_latency_tier=None,
        prefer_lower_cost=False,
        prefer_lower_latency=False,
        preferred_providers=None,
        preferred_cost_tier=None,
        preferred_latency_tier=None,
    )


def test_configuration_bootstrap_fails_before_execution_for_invalid_policy() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="classification_primary",
                provider="openai",
                provider_model="primary-provider-model",
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
        create_configured_llm_execution_service(settings)
