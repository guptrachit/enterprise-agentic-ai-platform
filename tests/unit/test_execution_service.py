import json
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from agent_platform.llm.errors import (
    LLMInvalidRequestError,
    LLMModelDisabledError,
    LLMTransientError,
)
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.execution_service import (
    LLMExecutionResult,
    LLMExecutionService,
)
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.routing_decision import RoutingDecision
from agent_platform.llm.routing_reason import (
    RoutingReason,
    RoutingReasonCode,
)
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
    workload: LLMWorkload = LLMWorkload.GENERAL,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider="openai",
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                workload,
            }
        ),
    )


def create_decision(
    *models: ModelDefinition,
) -> RoutingDecision:
    return RoutingDecision(
        selected_model=models[0],
        ranked_candidates=models,
        reasons=(
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message=(
                    f"Selected model '{models[0].name}' "
                    "as the highest-ranked eligible candidate"
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_execution_service_routes_and_executes_request() -> None:
    model = create_model(
        name="fast_general",
        workload=LLMWorkload.CLASSIFICATION,
    )

    router = Mock()
    router.route_decision.return_value = create_decision(model)

    client = AsyncMock()
    expected_response = object()
    client.generate.return_value = expected_response

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    result = await service.execute(
        LLMExecutionRequest(
            prompt="Classify this ticket.",
            workload=LLMWorkload.CLASSIFICATION,
        )
    )

    assert result is expected_response

    router.route_decision.assert_called_once_with(
        LLMWorkload.CLASSIFICATION,
        constraints=None,
        required_capabilities=frozenset(),
        preference=None,
    )


@pytest.mark.asyncio
async def test_execution_service_preserves_request_metadata() -> None:
    model = create_model(
        name="fast_general",
        workload=LLMWorkload.CLASSIFICATION,
    )

    router = Mock()
    router.route_decision.return_value = create_decision(model)

    client = AsyncMock()
    client.generate.return_value = object()

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    await service.execute(
        LLMExecutionRequest(
            prompt="Classify this ticket.",
            workload=LLMWorkload.CLASSIFICATION,
            correlation_id="corr-123",
            prompt_name="ticket_classifier",
            prompt_version="2.0",
        )
    )

    client.generate.assert_awaited_once_with(
        "Classify this ticket.",
        correlation_id="corr-123",
        prompt_name="ticket_classifier",
        prompt_version="2.0",
        workload="classification",
        logical_model="fast_general",
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
async def test_execution_service_does_not_create_client_when_routing_fails() -> None:
    router = Mock()
    router.route_decision.side_effect = LLMModelDisabledError("disabled_model")

    client_factory = Mock()

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    with pytest.raises(LLMModelDisabledError):
        await service.execute(
            LLMExecutionRequest(
                prompt="Classify this ticket.",
                workload=LLMWorkload.CLASSIFICATION,
            )
        )

    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_execution_service_falls_back_on_retryable_error() -> None:
    primary = create_model(
        name="primary",
        workload=LLMWorkload.CLASSIFICATION,
    )

    backup = create_model(
        name="backup",
        workload=LLMWorkload.CLASSIFICATION,
    )

    router = Mock()
    router.route_decision.return_value = create_decision(
        primary,
        backup,
    )

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMTransientError()

    backup_client = AsyncMock()
    expected_response = object()
    backup_client.generate.return_value = expected_response

    def client_factory(model: ModelDefinition):
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

    backup_client.generate.assert_awaited_once_with(
        "Classify this ticket.",
        correlation_id=None,
        prompt_name=None,
        prompt_version=None,
        workload="classification",
        logical_model="backup",
        fallback_used=True,
        fallback_from="primary",
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


@pytest.mark.asyncio
async def test_execution_service_does_not_fallback_on_non_retryable_error() -> None:
    primary = create_model(
        name="primary",
        workload=LLMWorkload.CLASSIFICATION,
    )

    backup = create_model(
        name="backup",
        workload=LLMWorkload.CLASSIFICATION,
    )

    router = Mock()
    router.route_decision.return_value = create_decision(
        primary,
        backup,
    )

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMInvalidRequestError()

    backup_client = AsyncMock()

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(
            side_effect=[
                primary_client,
                backup_client,
            ]
        ),
    )

    with pytest.raises(LLMInvalidRequestError):
        await service.execute(
            LLMExecutionRequest(
                prompt="Classify this ticket.",
                workload=LLMWorkload.CLASSIFICATION,
            )
        )

    backup_client.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_execution_service_passes_routing_constraints() -> None:
    model = create_model(
        name="fast_general",
        workload=LLMWorkload.CLASSIFICATION,
    )

    router = Mock()
    router.route_decision.return_value = create_decision(model)

    client = AsyncMock()
    client.generate.return_value = object()

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    constraints = RoutingConstraints(
        allowed_providers=frozenset(
            {
                "openai",
            }
        ),
        max_cost_tier=ModelCostTier.LOW,
        max_latency_tier=ModelLatencyTier.FAST,
    )

    await service.execute(
        LLMExecutionRequest(
            prompt="Classify this ticket.",
            workload=LLMWorkload.CLASSIFICATION,
            constraints=constraints,
        )
    )

    router.route_decision.assert_called_once_with(
        LLMWorkload.CLASSIFICATION,
        constraints=constraints,
        required_capabilities=frozenset(),
        preference=None,
    )


@pytest.mark.asyncio
async def test_execution_service_passes_required_capabilities() -> None:
    model = ModelDefinition(
        name="tool_capable",
        provider="openai",
        provider_model="tool-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.TOOL_CALLING,
            }
        ),
    )

    router = Mock()
    router.route_decision.return_value = create_decision(model)

    client = AsyncMock()
    client.generate.return_value = object()

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    required = frozenset(
        {
            ModelCapability.TOOL_CALLING,
        }
    )

    await service.execute(
        LLMExecutionRequest(
            prompt="Use a tool.",
            required_capabilities=required,
        )
    )

    router.route_decision.assert_called_once_with(
        LLMWorkload.GENERAL,
        constraints=None,
        required_capabilities=required,
        preference=None,
    )


@pytest.mark.asyncio
async def test_execution_service_passes_model_preference() -> None:
    model = create_model(
        name="preferred_model",
    )

    router = Mock()
    router.route_decision.return_value = create_decision(model)

    client = AsyncMock()
    client.generate.return_value = object()

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    preference = ModelPreference(
        prefer_lower_cost=True,
        prefer_lower_latency=True,
        preferred_providers=(
            "openai",
            "anthropic",
        ),
        preferred_cost_tier=ModelCostTier.LOW,
        preferred_latency_tier=ModelLatencyTier.FAST,
    )

    await service.execute(
        LLMExecutionRequest(
            prompt="Answer this question.",
            preference=preference,
        )
    )

    router.route_decision.assert_called_once_with(
        LLMWorkload.GENERAL,
        constraints=None,
        required_capabilities=frozenset(),
        preference=preference,
    )


@pytest.mark.asyncio
async def test_execute_with_decision_returns_routing_decision() -> None:
    model = create_model(
        name="selected_model",
    )

    decision = RoutingDecision(
        selected_model=model,
        ranked_candidates=(model,),
        reasons=(
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message="Selected model 'selected_model'",
            ),
        ),
    )

    router = Mock()
    router.route_decision.return_value = decision

    client = AsyncMock()
    expected_response = object()
    client.generate.return_value = expected_response

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    result = await service.execute_with_decision(
        LLMExecutionRequest(
            prompt="Answer this question.",
        )
    )

    assert isinstance(result, LLMExecutionResult)
    assert result.response is expected_response
    assert result.routing_decision is decision
    assert result.executed_model is model
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_execute_with_decision_distinguishes_fallback_model() -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    decision = RoutingDecision(
        selected_model=primary,
        ranked_candidates=(
            primary,
            backup,
        ),
        reasons=(
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message="Primary model selected",
            ),
        ),
    )

    router = Mock()
    router.route_decision.return_value = decision

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMTransientError()

    backup_client = AsyncMock()
    expected_response = object()
    backup_client.generate.return_value = expected_response

    def client_factory(model: ModelDefinition):
        if model is primary:
            return primary_client

        return backup_client

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    result = await service.execute_with_decision(
        LLMExecutionRequest(
            prompt="Answer this question.",
        )
    )

    assert result.response is expected_response
    assert result.routing_decision.selected_model is primary
    assert result.executed_model is backup
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_execution_service_logs_routing_decision(
    caplog,
) -> None:
    selected = create_model(
        name="selected",
    )

    decision = RoutingDecision(
        selected_model=selected,
        ranked_candidates=(selected,),
        rejected_models=("disabled",),
        reasons=(
            RoutingReason(
                code=RoutingReasonCode.DISABLED,
                message=("Model 'disabled' rejected because it is disabled"),
            ),
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message="Selected model 'selected'",
            ),
        ),
    )

    router = Mock()
    router.route_decision.return_value = decision

    client = AsyncMock()
    client.generate.return_value = object()

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        await service.execute(
            LLMExecutionRequest(
                prompt="Answer.",
                correlation_id="corr-routing-001",
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

    assert payload["selected_model"] == "selected"
    assert payload["executed_model"] == "selected"

    assert payload["ranked_candidates"] == [
        "selected",
    ]

    assert payload["rejected_models"] == [
        "disabled",
    ]

    assert payload["routing_reason_codes"] == [
        "disabled",
        "selected",
    ]

    assert payload["routing_reasons"] == [
        "Model 'disabled' rejected because it is disabled",
        "Selected model 'selected'",
    ]

    assert payload["fallback_used"] is False
    assert payload["success"] is True
    assert payload["correlation_id"] == "corr-routing-001"


@pytest.mark.asyncio
async def test_execution_service_logs_fallback_routing_decision(
    caplog,
) -> None:
    primary = create_model(
        name="primary",
    )

    backup = create_model(
        name="backup",
    )

    decision = RoutingDecision(
        selected_model=primary,
        ranked_candidates=(
            primary,
            backup,
        ),
        reasons=(
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message="Primary selected",
            ),
        ),
    )

    router = Mock()
    router.route_decision.return_value = decision

    primary_client = AsyncMock()
    primary_client.generate.side_effect = LLMTransientError()

    backup_client = AsyncMock()
    backup_client.generate.return_value = object()

    def client_factory(model: ModelDefinition):
        if model is primary:
            return primary_client

        return backup_client

    service = LLMExecutionService(
        router=router,
        client_factory=client_factory,
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        await service.execute(
            LLMExecutionRequest(
                prompt="Answer.",
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

    assert payload["selected_model"] == "primary"
    assert payload["executed_model"] == "backup"

    assert payload["routing_reason_codes"] == [
        "selected",
    ]

    assert payload["routing_reasons"] == [
        "Primary selected",
    ]

    assert payload["fallback_used"] is True
    assert payload["success"] is True


@pytest.mark.asyncio
async def test_execution_service_logs_failed_routing_decision(
    caplog,
) -> None:
    primary = create_model(
        name="primary",
    )

    decision = RoutingDecision(
        selected_model=primary,
        ranked_candidates=(primary,),
        reasons=(
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message="Primary selected",
            ),
        ),
    )

    router = Mock()
    router.route_decision.return_value = decision

    client = AsyncMock()
    client.generate.side_effect = LLMInvalidRequestError()

    service = LLMExecutionService(
        router=router,
        client_factory=Mock(return_value=client),
    )

    with (
        caplog.at_level(
            logging.INFO,
            logger="agent_platform.llm",
        ),
        pytest.raises(LLMInvalidRequestError),
    ):
        await service.execute(
            LLMExecutionRequest(
                prompt="Invalid request.",
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

    assert payload["selected_model"] == "primary"
    assert payload["executed_model"] == "primary"

    assert payload["routing_reason_codes"] == [
        "selected",
    ]

    assert payload["routing_reasons"] == [
        "Primary selected",
    ]

    assert payload["success"] is False
    assert payload["fallback_used"] is False

    assert payload["error_type"] == "LLMInvalidRequestError"
