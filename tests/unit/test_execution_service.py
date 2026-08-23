from unittest.mock import AsyncMock, Mock

import pytest

from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.execution_service import LLMExecutionService
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.workload import LLMWorkload


@pytest.mark.asyncio
async def test_execution_service_routes_and_executes_request() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = Mock()
    router.route.return_value = model

    client = AsyncMock()

    client_factory = Mock(return_value=client)

    expected_response = object()
    client.generate.return_value = expected_response

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    request = LLMExecutionRequest(
        prompt="Classify this ticket.",
        workload=LLMWorkload.CLASSIFICATION,
    )

    result = await service.execute(request)

    assert result is expected_response

    router.route.assert_called_once_with(LLMWorkload.CLASSIFICATION)

    client_factory.assert_called_once_with(model)

    client.generate.assert_awaited_once_with(
        "Classify this ticket.",
        correlation_id=None,
        prompt_name=None,
        prompt_version=None,
        workload="classification",
        logical_model="fast_general",
    )


@pytest.mark.asyncio
async def test_execution_service_preserves_request_metadata() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = Mock()
    router.route.return_value = model

    client = AsyncMock()
    client_factory = Mock(return_value=client)

    expected_response = object()
    client.generate.return_value = expected_response

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    request = LLMExecutionRequest(
        prompt="Classify this ticket.",
        workload=LLMWorkload.CLASSIFICATION,
        correlation_id="corr-123",
        prompt_name="ticket_classifier",
        prompt_version="2.0",
    )

    result = await service.execute(request)

    assert result is expected_response

    client.generate.assert_awaited_once_with(
        "Classify this ticket.",
        correlation_id="corr-123",
        prompt_name="ticket_classifier",
        prompt_version="2.0",
        workload="classification",
        logical_model="fast_general",
    )


@pytest.mark.asyncio
async def test_execution_service_does_not_create_client_when_routing_fails() -> None:
    from agent_platform.llm.errors import LLMModelDisabledError

    router = Mock()
    router.route.side_effect = LLMModelDisabledError("disabled_model")

    client_factory = Mock()

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    request = LLMExecutionRequest(
        prompt="Classify this ticket.",
        workload=LLMWorkload.CLASSIFICATION,
    )

    with pytest.raises(LLMModelDisabledError):
        await service.execute(request)

    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_execution_service_falls_back_on_retryable_error() -> None:
    from agent_platform.llm.errors import LLMTransientError

    primary = ModelDefinition(
        name="primary",
        provider="openai",
        provider_model="primary-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    backup = ModelDefinition(
        name="backup",
        provider="openai",
        provider_model="backup-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = Mock()
    router.route_candidates.return_value = (
        primary,
        backup,
    )

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMTransientError()

    backup_client = AsyncMock()
    expected_response = object()
    backup_client.generate.return_value = expected_response

    def client_factory(model):
        if model is primary:
            return primary_client

        return backup_client

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    result = await service.execute(
        LLMExecutionRequest(
            prompt="Classify this ticket.",
            workload=LLMWorkload.CLASSIFICATION,
        )
    )

    assert result is expected_response


@pytest.mark.asyncio
async def test_execution_service_does_not_fallback_on_non_retryable_error() -> None:
    from agent_platform.llm.errors import LLMInvalidRequestError

    primary = ModelDefinition(
        name="primary",
        provider="openai",
        provider_model="primary-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    backup = ModelDefinition(
        name="backup",
        provider="openai",
        provider_model="backup-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = Mock()
    router.route_candidates.return_value = (
        primary,
        backup,
    )

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMInvalidRequestError()

    backup_client = AsyncMock()

    client_factory = Mock(
        side_effect=[
            primary_client,
            backup_client,
        ]
    )

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    with pytest.raises(LLMInvalidRequestError):
        await service.execute(
            LLMExecutionRequest(
                prompt="Classify this ticket.",
                workload=LLMWorkload.CLASSIFICATION,
            )
        )

    assert client_factory.call_count == 1
    backup_client.generate.assert_not_awaited()
