from unittest.mock import AsyncMock, Mock

import pytest

from agent_platform.llm.api_models import (
    LLMGenerateRequest,
)
from agent_platform.llm.api_service import (
    GovernedLLMAPIService,
)
from agent_platform.llm.execution_service import (
    LLMExecutionResult,
)
from agent_platform.llm.model_definition import (
    ModelDefinition,
)
from agent_platform.llm.workload import LLMWorkload


def create_model() -> ModelDefinition:
    return ModelDefinition(
        name="primary",
        provider="openai",
        provider_model="gpt-test",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )


@pytest.mark.asyncio
async def test_api_service_translates_request_to_execution() -> None:
    model = create_model()

    response = Mock()
    response.content = "Generated answer"

    execution_result = LLMExecutionResult(
        response=response,
        routing_decision=Mock(),
        executed_model=model,
        fallback_used=False,
    )

    runtime = Mock()
    runtime.policy_identifier = "production-routing-policy@1.0.0"
    runtime.execute_with_decision = AsyncMock(return_value=execution_result)

    service = GovernedLLMAPIService(runtime=runtime)

    request = LLMGenerateRequest(
        prompt="Answer this.",
        correlation_id="corr-001",
        prompt_name="general-prompt",
        prompt_version="1.0",
    )

    result = await service.generate(request)

    runtime.execute_with_decision.assert_awaited_once()

    execution_request = runtime.execute_with_decision.await_args.args[0]

    assert execution_request.prompt == "Answer this."

    assert execution_request.workload is LLMWorkload.GENERAL

    assert execution_request.correlation_id == "corr-001"

    assert execution_request.prompt_name == ("general-prompt")

    assert execution_request.prompt_version == "1.0"

    assert result.content == "Generated answer"

    assert result.policy_identifier == ("production-routing-policy@1.0.0")

    assert result.model == "primary"
    assert result.provider == "openai"


@pytest.mark.asyncio
async def test_api_service_preserves_requested_workload() -> None:
    model = ModelDefinition(
        name="classifier",
        provider="openai",
        provider_model="classifier-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    response = Mock()
    response.content = "classified"

    runtime = Mock()
    runtime.policy_identifier = "production-routing-policy@2.0.0"

    runtime.execute_with_decision = AsyncMock(
        return_value=LLMExecutionResult(
            response=response,
            routing_decision=Mock(),
            executed_model=model,
            fallback_used=False,
        )
    )

    service = GovernedLLMAPIService(runtime=runtime)

    await service.generate(
        LLMGenerateRequest(
            prompt="Classify this.",
            workload=LLMWorkload.CLASSIFICATION,
        )
    )

    execution_request = runtime.execute_with_decision.await_args.args[0]

    assert execution_request.workload is LLMWorkload.CLASSIFICATION


@pytest.mark.asyncio
async def test_api_service_uses_executed_fallback_model() -> None:
    backup = create_model()

    response = Mock()
    response.content = "Fallback answer"

    runtime = Mock()
    runtime.policy_identifier = "production-routing-policy@1.0.0"

    runtime.execute_with_decision = AsyncMock(
        return_value=LLMExecutionResult(
            response=response,
            routing_decision=Mock(),
            executed_model=backup,
            fallback_used=True,
        )
    )

    service = GovernedLLMAPIService(runtime=runtime)

    result = await service.generate(
        LLMGenerateRequest(
            prompt="Answer.",
        )
    )

    assert result.content == "Fallback answer"
    assert result.model == "primary"
    assert result.provider == "openai"


@pytest.mark.asyncio
async def test_api_service_allows_unresolved_policy_identifier() -> None:
    model = create_model()

    response = Mock()
    response.content = "Answer"

    runtime = Mock()
    runtime.policy_identifier = None

    runtime.execute_with_decision = AsyncMock(
        return_value=LLMExecutionResult(
            response=response,
            routing_decision=Mock(),
            executed_model=model,
            fallback_used=False,
        )
    )

    service = GovernedLLMAPIService(runtime=runtime)

    result = await service.generate(
        LLMGenerateRequest(
            prompt="Answer.",
        )
    )

    assert result.policy_identifier is None
