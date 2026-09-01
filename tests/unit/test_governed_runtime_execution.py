import json
import logging
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


def create_governed_policy(
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


@pytest.mark.asyncio
async def test_governed_runtime_execution_follows_active_policy(
    caplog,
) -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    registry = RoutingPolicyRegistry()

    old_policy = create_governed_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        selected_model="primary",
    )

    new_policy = create_governed_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
        selected_model="backup",
    )

    registry.register(old_policy)
    registry.register(new_policy)

    primary_client = AsyncMock()
    primary_response = object()
    primary_client.generate.return_value = primary_response

    backup_client = AsyncMock()
    backup_response = object()
    backup_client.generate.return_value = backup_response

    def client_factory(
        model: ModelDefinition,
    ):
        if model.name == "primary":
            return primary_client

        return backup_client

    router_factory = RuntimeModelRouterFactory(
        resolver=ActiveRoutingPolicyResolver(registry),
        models={
            "primary": primary,
            "backup": backup,
        },
    )

    execution_factory = RuntimeLLMExecutionServiceFactory(
        router_factory=router_factory,
        client_factory=client_factory,
    )

    before_service = execution_factory.create("production-routing-policy")

    before_result = await before_service.execute(
        LLMExecutionRequest(
            prompt="Before activation.",
            correlation_id="corr-before",
        )
    )

    assert before_result is primary_response

    assert before_service.policy_identifier == ("production-routing-policy@1.0.0")

    primary_client.generate.assert_awaited_once()
    backup_client.generate.assert_not_awaited()

    registry.activate(new_policy.identifier)

    after_service = execution_factory.create("production-routing-policy")

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        after_result = await after_service.execute(
            LLMExecutionRequest(
                prompt="After activation.",
                correlation_id="corr-after",
            )
        )

    assert after_result is backup_response

    assert after_service.policy_identifier == ("production-routing-policy@1.1.0")

    backup_client.generate.assert_awaited_once()

    active = registry.get_active("production-routing-policy")

    assert active is not None

    assert active.identifier == ("production-routing-policy@1.1.0")

    assert (
        registry.get(old_policy.identifier).status
        is RoutingPolicyLifecycleStatus.RETIRED
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
    assert payload["success"] is True


@pytest.mark.asyncio
async def test_existing_service_keeps_resolved_policy_snapshot() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    registry = RoutingPolicyRegistry()

    old_policy = create_governed_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
        selected_model="primary",
    )

    new_policy = create_governed_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
        selected_model="backup",
    )

    registry.register(old_policy)
    registry.register(new_policy)

    primary_client = AsyncMock()
    primary_response = object()
    primary_client.generate.return_value = primary_response

    backup_client = AsyncMock()
    backup_response = object()
    backup_client.generate.return_value = backup_response

    def client_factory(
        model: ModelDefinition,
    ):
        if model.name == "primary":
            return primary_client

        return backup_client

    execution_factory = RuntimeLLMExecutionServiceFactory(
        router_factory=RuntimeModelRouterFactory(
            resolver=ActiveRoutingPolicyResolver(registry),
            models={
                "primary": primary,
                "backup": backup,
            },
        ),
        client_factory=client_factory,
    )

    old_service = execution_factory.create("production-routing-policy")

    registry.activate(new_policy.identifier)

    result = await old_service.execute(
        LLMExecutionRequest(
            prompt="Existing service request.",
        )
    )

    assert result is primary_response

    assert old_service.policy_identifier == ("production-routing-policy@1.0.0")
