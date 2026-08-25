import json
import logging
from unittest.mock import AsyncMock

import pytest

from agent_platform.config import Settings
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.settings_driven_governed_runtime import (
    create_settings_driven_governed_runtime,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider="openai",
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )


def create_policy(
    *,
    version: str,
    status: RoutingPolicyLifecycleStatus,
    selected_model: str,
    name: str = "production-routing-policy",
) -> GovernedRoutingPolicy:
    return GovernedRoutingPolicy(
        versioned_policy=VersionedRoutingPolicy(
            policy=ModelPolicy(
                assignments={
                    LLMWorkload.GENERAL: selected_model,
                }
            ),
            metadata=RoutingPolicyMetadata(
                name=name,
                version=RoutingPolicyVersion.parse(version),
            ),
        ),
        status=status,
    )


@pytest.mark.asyncio
async def test_settings_driven_runtime_executes_active_policy(
    caplog,
) -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    primary_client = AsyncMock()
    primary_response = object()
    primary_client.generate.return_value = primary_response

    backup_client = AsyncMock()
    backup_response = object()
    backup_client.generate.return_value = backup_response

    registry = RoutingPolicyRegistry()

    active_policy = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        selected_model="primary",
    )

    registry.register(active_policy)

    def client_factory(
        model: ModelDefinition,
    ):
        if model.name == "primary":
            return primary_client

        return backup_client

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
        ),
        registry=registry,
        models={
            "primary": primary,
            "backup": backup,
        },
        client_factory=client_factory,
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        result = await runtime.refresh_service.execute(
            LLMExecutionRequest(
                prompt="Answer.",
                correlation_id="corr-settings-001",
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
    assert payload["correlation_id"] == "corr-settings-001"


@pytest.mark.asyncio
async def test_settings_driven_per_request_mode_adopts_new_policy() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    primary_client = AsyncMock()
    primary_response = object()
    primary_client.generate.return_value = primary_response

    backup_client = AsyncMock()
    backup_response = object()
    backup_client.generate.return_value = backup_response

    registry = RoutingPolicyRegistry()

    old_policy = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        selected_model="primary",
    )

    new_policy = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
        selected_model="backup",
    )

    registry.register(old_policy)
    registry.register(new_policy)

    def client_factory(
        model: ModelDefinition,
    ):
        if model.name == "primary":
            return primary_client

        return backup_client

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.PER_REQUEST),
        ),
        registry=registry,
        models={
            "primary": primary,
            "backup": backup,
        },
        client_factory=client_factory,
    )

    first = await runtime.refresh_service.execute(
        LLMExecutionRequest(
            prompt="Before activation.",
        )
    )

    assert first is primary_response

    assert runtime.refresh_service.policy_identifier == (
        "production-routing-policy@1.0.0"
    )

    registry.activate(new_policy.identifier)

    second = await runtime.refresh_service.execute(
        LLMExecutionRequest(
            prompt="After activation.",
        )
    )

    assert second is backup_response

    assert runtime.refresh_service.policy_identifier == (
        "production-routing-policy@1.1.0"
    )

    primary_client.generate.assert_awaited_once()
    backup_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_settings_driven_custom_policy_name_executes_correct_policy() -> None:
    model = create_model(
        name="enterprise_model",
    )

    client = AsyncMock()
    expected_response = object()
    client.generate.return_value = expected_response

    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            name="enterprise-routing-policy",
            version="2.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            selected_model="enterprise_model",
        )
    )

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
            llm_routing_policy_name=("enterprise-routing-policy"),
        ),
        registry=registry,
        models={
            "enterprise_model": model,
        },
        client_factory=lambda _: client,
    )

    result = await runtime.refresh_service.execute(
        LLMExecutionRequest(
            prompt="Enterprise request.",
        )
    )

    assert result is expected_response

    assert runtime.refresh_service.policy_identifier == (
        "enterprise-routing-policy@2.0.0"
    )


@pytest.mark.asyncio
async def test_settings_driven_runtime_updates_refresh_metrics() -> None:
    primary = create_model(
        name="primary",
    )

    client = AsyncMock()
    client.generate.return_value = object()

    registry = RoutingPolicyRegistry()

    registry.register(
        create_policy(
            version="1.0.0",
            status=RoutingPolicyLifecycleStatus.ACTIVE,
            selected_model="primary",
        )
    )

    runtime = create_settings_driven_governed_runtime(
        settings=Settings(
            openai_api_key="test-key",
            llm_routing_policy_refresh_mode=(RuntimePolicyRefreshMode.PER_REQUEST),
        ),
        registry=registry,
        models={
            "primary": primary,
        },
        client_factory=lambda _: client,
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
