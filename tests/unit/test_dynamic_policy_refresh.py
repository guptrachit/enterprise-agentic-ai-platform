from unittest.mock import AsyncMock

import pytest

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)
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
from agent_platform.llm.runtime_llm_execution_service_factory import (
    RuntimeLLMExecutionServiceFactory,
)
from agent_platform.llm.runtime_model_router_factory import (
    RuntimeModelRouterFactory,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshConfig,
    RuntimePolicyRefreshMode,
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
) -> GovernedRoutingPolicy:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: selected_model,
        }
    )

    return GovernedRoutingPolicy(
        versioned_policy=VersionedRoutingPolicy(
            policy=policy,
            metadata=RoutingPolicyMetadata(
                name="production-routing-policy",
                version=RoutingPolicyVersion.parse(version),
            ),
        ),
        status=status,
    )


def create_execution_factory(
    *,
    registry: RoutingPolicyRegistry,
    models: dict[str, ModelDefinition],
    clients: dict[str, AsyncMock],
) -> RuntimeLLMExecutionServiceFactory:
    def client_factory(
        model: ModelDefinition,
    ):
        return clients[model.name]

    return RuntimeLLMExecutionServiceFactory(
        router_factory=RuntimeModelRouterFactory(
            resolver=ActiveRoutingPolicyResolver(registry),
            models=models,
        ),
        client_factory=client_factory,
    )


@pytest.mark.asyncio
async def test_process_static_keeps_original_policy_after_activation() -> None:
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

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-routing-policy",
        factory=create_execution_factory(
            registry=registry,
            models={
                "primary": primary,
                "backup": backup,
            },
            clients={
                "primary": primary_client,
                "backup": backup_client,
            },
        ),
    )

    registry.activate(new_policy.identifier)

    result = await wrapper.execute(
        LLMExecutionRequest(
            prompt="Use static policy.",
        )
    )

    assert result is primary_response

    assert wrapper.policy_identifier == ("production-routing-policy@1.0.0")

    primary_client.generate.assert_awaited_once()
    backup_client.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_per_request_refresh_uses_newly_activated_policy() -> None:
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

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-routing-policy",
        factory=create_execution_factory(
            registry=registry,
            models={
                "primary": primary,
                "backup": backup,
            },
            clients={
                "primary": primary_client,
                "backup": backup_client,
            },
        ),
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
    )

    first = await wrapper.execute(
        LLMExecutionRequest(
            prompt="Before activation.",
        )
    )

    assert first is primary_response

    assert wrapper.policy_identifier == ("production-routing-policy@1.0.0")

    registry.activate(new_policy.identifier)

    second = await wrapper.execute(
        LLMExecutionRequest(
            prompt="After activation.",
        )
    )

    assert second is backup_response

    assert wrapper.policy_identifier == ("production-routing-policy@1.1.0")

    primary_client.generate.assert_awaited_once()
    backup_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_ttl_refresh_changes_policy_only_after_expiration() -> None:
    now = [100.0]

    def clock() -> float:
        return now[0]

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

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-routing-policy",
        factory=create_execution_factory(
            registry=registry,
            models={
                "primary": primary,
                "backup": backup,
            },
            clients={
                "primary": primary_client,
                "backup": backup_client,
            },
        ),
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=60.0,
        ),
        clock=clock,
    )

    first = await wrapper.execute(
        LLMExecutionRequest(
            prompt="Initial request.",
        )
    )

    assert first is primary_response

    assert wrapper.policy_identifier == ("production-routing-policy@1.0.0")

    registry.activate(new_policy.identifier)

    now[0] = 130.0

    before_expiry = await wrapper.execute(
        LLMExecutionRequest(
            prompt="Before expiry.",
        )
    )

    assert before_expiry is primary_response

    assert wrapper.policy_identifier == ("production-routing-policy@1.0.0")

    now[0] = 160.0

    after_expiry = await wrapper.execute(
        LLMExecutionRequest(
            prompt="After expiry.",
        )
    )

    assert after_expiry is backup_response

    assert wrapper.policy_identifier == ("production-routing-policy@1.1.0")

    assert primary_client.generate.await_count == 2
    backup_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_per_request_telemetry_uses_refreshed_policy_identifier(
    caplog,
) -> None:
    import json
    import logging

    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    primary_client = AsyncMock()
    primary_client.generate.return_value = object()

    backup_client = AsyncMock()
    backup_client.generate.return_value = object()

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

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-routing-policy",
        factory=create_execution_factory(
            registry=registry,
            models={
                "primary": primary,
                "backup": backup,
            },
            clients={
                "primary": primary_client,
                "backup": backup_client,
            },
        ),
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Before.",
            correlation_id="corr-before",
        )
    )

    registry.activate(new_policy.identifier)

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        await wrapper.execute(
            LLMExecutionRequest(
                prompt="After.",
                correlation_id="corr-after",
            )
        )

    routing_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_routing_decision ")
    ]

    assert len(routing_records) == 1

    payload = json.loads(
        routing_records[0].getMessage().removeprefix("llm_routing_decision ")
    )

    assert payload["policy_identifier"] == ("production-routing-policy@1.1.0")

    assert payload["selected_model"] == "backup"
    assert payload["executed_model"] == "backup"
    assert payload["correlation_id"] == "corr-after"
