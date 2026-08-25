import json
import logging
from unittest.mock import AsyncMock

import pytest

from agent_platform.config import Settings
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.fully_configured_governed_runtime import (
    create_fully_configured_governed_runtime,
)
from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.workload import LLMWorkload


def create_settings(
    *,
    refresh_mode: RuntimePolicyRefreshMode = (RuntimePolicyRefreshMode.PROCESS_STATIC),
) -> Settings:
    return Settings(
        openai_api_key="test-key",
        llm_models=(
            ModelConfig(
                name="primary",
                provider="openai",
                provider_model="primary-provider-model",
                workloads=(LLMWorkload.GENERAL,),
            ),
            ModelConfig(
                name="backup",
                provider="openai",
                provider_model="backup-provider-model",
                workloads=(LLMWorkload.GENERAL,),
            ),
        ),
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.GENERAL: (
                    "primary",
                    "backup",
                ),
            }
        ),
        llm_routing_policy_refresh_mode=refresh_mode,
    )


@pytest.mark.asyncio
async def test_fully_configured_runtime_executes_from_settings(
    caplog,
) -> None:
    primary_client = AsyncMock()
    primary_response = object()
    primary_client.generate.return_value = primary_response

    backup_client = AsyncMock()
    backup_client.generate.return_value = object()

    def client_factory(model):
        if model.name == "primary":
            return primary_client

        return backup_client

    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(),
        client_factory=client_factory,
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        result = await runtime.refresh_service.execute(
            LLMExecutionRequest(
                prompt="Answer from governed runtime.",
                correlation_id="corr-full-runtime-001",
            )
        )

    assert result is primary_response

    assert runtime.refresh_service.policy_identifier == (
        "production-routing-policy@1.0.0"
    )

    primary_client.generate.assert_awaited_once()
    backup_client.generate.assert_not_awaited()

    routing_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_routing_decision ")
    ]

    assert len(routing_records) == 1

    payload = json.loads(
        routing_records[0].getMessage().removeprefix("llm_routing_decision ")
    )

    assert payload["policy_identifier"] == ("production-routing-policy@1.0.0")

    assert payload["selected_model"] == "primary"
    assert payload["executed_model"] == "primary"

    assert payload["correlation_id"] == ("corr-full-runtime-001")

    assert payload["success"] is True


@pytest.mark.asyncio
async def test_fully_configured_runtime_preserves_fallback() -> None:
    from agent_platform.llm.errors import LLMTransientError

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMTransientError()

    backup_client = AsyncMock()
    backup_response = object()
    backup_client.generate.return_value = backup_response

    def client_factory(model):
        if model.name == "primary":
            return primary_client

        return backup_client

    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(),
        client_factory=client_factory,
    )

    result = await runtime.refresh_service.execute(
        LLMExecutionRequest(
            prompt="Use fallback.",
        )
    )

    assert result is backup_response

    primary_client.generate.assert_awaited_once()
    backup_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_fully_configured_per_request_runtime_executes() -> None:
    primary_client = AsyncMock()
    response = object()
    primary_client.generate.return_value = response

    backup_client = AsyncMock()

    def client_factory(model):
        if model.name == "primary":
            return primary_client

        return backup_client

    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(
            refresh_mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
        client_factory=client_factory,
    )

    assert runtime.refresh_service.policy_identifier is None

    result = await runtime.refresh_service.execute(
        LLMExecutionRequest(
            prompt="Resolve lazily.",
        )
    )

    assert result is response

    assert runtime.refresh_service.policy_identifier == (
        "production-routing-policy@1.0.0"
    )


@pytest.mark.asyncio
async def test_fully_configured_runtime_updates_refresh_metrics() -> None:
    primary_client = AsyncMock()
    primary_client.generate.return_value = object()

    backup_client = AsyncMock()

    def client_factory(model):
        if model.name == "primary":
            return primary_client

        return backup_client

    runtime = create_fully_configured_governed_runtime(
        settings=create_settings(
            refresh_mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
        client_factory=client_factory,
    )

    await runtime.refresh_service.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    await runtime.refresh_service.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    assert runtime.refresh_metrics is not None

    snapshot = runtime.refresh_metrics.snapshot()

    assert snapshot.total_resolutions == 2
    assert snapshot.refreshes == 2
    assert snapshot.cache_hits == 0
