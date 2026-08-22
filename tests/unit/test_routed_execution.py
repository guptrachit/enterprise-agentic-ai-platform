import pytest

from agent_platform.config import Settings
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.factory import create_llm_execution_service
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.openai_client import OpenAIClient
from agent_platform.llm.workload import LLMWorkload


@pytest.mark.asyncio
async def test_routed_execution_uses_model_selected_by_policy(
    monkeypatch,
) -> None:
    selected_model: str | None = None
    captured_prompt: str | None = None

    async def fake_generate(
        self,
        prompt: str,
        *,
        correlation_id: str | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
    ):
        nonlocal selected_model
        nonlocal captured_prompt

        selected_model = self.model
        captured_prompt = prompt

        return object()

    monkeypatch.setattr(
        OpenAIClient,
        "generate",
        fake_generate,
    )

    model = ModelDefinition(
        name="classification_model",
        provider="openai",
        provider_model="routed-classification-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    policy = ModelPolicy(
        assignments={
            LLMWorkload.CLASSIFICATION: "classification_model",
        }
    )

    router = ModelRouter(
        models={
            "classification_model": model,
        },
        policy=policy,
    )

    settings = Settings(
        openai_api_key="test-key",
        llm_model="default-model",
    )

    service = create_llm_execution_service(
        settings,
        router,
    )

    request = LLMExecutionRequest(
        prompt="Classify this ticket.",
        workload=LLMWorkload.CLASSIFICATION,
    )

    result = await service.execute(request)

    assert result is not None
    assert selected_model == "routed-classification-model"
    assert captured_prompt == "Classify this ticket."


@pytest.mark.asyncio
async def test_routed_execution_preserves_execution_metadata(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_generate(
        self,
        prompt: str,
        *,
        correlation_id: str | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
    ):
        captured["model"] = self.model
        captured["prompt"] = prompt
        captured["correlation_id"] = correlation_id
        captured["prompt_name"] = prompt_name
        captured["prompt_version"] = prompt_version

        return object()

    monkeypatch.setattr(
        OpenAIClient,
        "generate",
        fake_generate,
    )

    model = ModelDefinition(
        name="classification_model",
        provider="openai",
        provider_model="routed-classification-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = ModelRouter(
        models={
            "classification_model": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: "classification_model",
            }
        ),
    )

    service = create_llm_execution_service(
        Settings(
            openai_api_key="test-key",
        ),
        router,
    )

    await service.execute(
        LLMExecutionRequest(
            prompt="Classify this ticket.",
            workload=LLMWorkload.CLASSIFICATION,
            correlation_id="corr-456",
            prompt_name="ticket_classifier",
            prompt_version="2.0",
        )
    )

    assert captured == {
        "model": "routed-classification-model",
        "prompt": "Classify this ticket.",
        "correlation_id": "corr-456",
        "prompt_name": "ticket_classifier",
        "prompt_version": "2.0",
    }
