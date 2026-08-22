from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from agent_platform.llm.prompt import PromptTemplate
from agent_platform.llm.prompt_registry import PromptRegistry
from agent_platform.llm.prompt_service import PromptExecutionService


@pytest.mark.asyncio
async def test_generate_active_uses_active_prompt_version() -> None:
    registry = PromptRegistry()

    version_1 = PromptTemplate(
        name="ticket_classifier",
        version="1.0",
        template="Version 1: {ticket_text}",
    )

    version_2 = PromptTemplate(
        name="ticket_classifier",
        version="2.0",
        template="Version 2: {ticket_text}",
    )

    registry.register(version_1)
    registry.register(version_2)
    registry.set_active(
        "ticket_classifier",
        "2.0",
    )

    client = AsyncMock()
    expected_response = object()

    client.generate_from_template.return_value = expected_response

    service = PromptExecutionService(
        registry,
        client,
    )

    result = await service.generate_active(
        "ticket_classifier",
        {
            "ticket_text": "My card was charged twice.",
        },
        correlation_id="corr-123",
    )

    assert result is expected_response

    client.generate_from_template.assert_awaited_once_with(
        version_2,
        {
            "ticket_text": "My card was charged twice.",
        },
        correlation_id="corr-123",
    )


class ClassificationResult(BaseModel):
    category: str
    confidence: float


@pytest.mark.asyncio
async def test_generate_structured_active_uses_active_prompt_version() -> None:
    registry = PromptRegistry()

    prompt = PromptTemplate(
        name="ticket_classifier",
        version="2.0",
        template="Classify this ticket: {ticket_text}",
    )

    registry.register(prompt)
    registry.set_active(
        "ticket_classifier",
        "2.0",
    )

    client = AsyncMock()
    expected_response = object()

    client.generate_structured_from_template.return_value = expected_response

    service = PromptExecutionService(
        registry,
        client,
    )

    result = await service.generate_structured_active(
        "ticket_classifier",
        ClassificationResult,
        {
            "ticket_text": "My card was charged twice.",
        },
        correlation_id="corr-structured-123",
    )

    assert result is expected_response

    client.generate_structured_from_template.assert_awaited_once_with(
        prompt,
        ClassificationResult,
        {
            "ticket_text": "My card was charged twice.",
        },
        correlation_id="corr-structured-123",
    )


@pytest.mark.asyncio
async def test_generate_active_fails_before_llm_when_active_not_set() -> None:
    from agent_platform.llm.errors import LLMPromptActiveVersionNotSetError

    registry = PromptRegistry()

    registry.register(
        PromptTemplate(
            name="ticket_classifier",
            version="1.0",
            template="Classify: {ticket_text}",
        )
    )

    client = AsyncMock()

    service = PromptExecutionService(
        registry,
        client,
    )

    with pytest.raises(LLMPromptActiveVersionNotSetError):
        await service.generate_active(
            "ticket_classifier",
            {
                "ticket_text": "Payment failed.",
            },
        )

    client.generate_from_template.assert_not_awaited()
