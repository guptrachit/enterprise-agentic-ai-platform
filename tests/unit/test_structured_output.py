from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_platform.config import Settings
from agent_platform.llm.errors import (
    LLMRefusalError,
    LLMStructuredParseError,
    LLMStructuredValidationError,
)
from agent_platform.llm.openai_client import OpenAIClient


class ClassificationResult(BaseModel):
    category: str
    confidence: float


class FakeUsage:
    input_tokens = 20
    output_tokens = 10


class FakeStructuredResponse:
    id = "resp_structured_123"
    usage = FakeUsage()
    output_parsed = ClassificationResult(
        category="technical",
        confidence=0.95,
    )


@pytest.mark.asyncio
async def test_generate_structured_returns_validated_model() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    async def fake_parse(
        *,
        model: str,
        input: str,
        text_format: type[BaseModel],
    ) -> FakeStructuredResponse:
        return FakeStructuredResponse()

    client.client.responses.parse = fake_parse

    response = await client.generate_structured(
        "Classify this request.",
        ClassificationResult,
        correlation_id="corr-structured-123",
    )

    assert isinstance(response.parsed, ClassificationResult)
    assert response.parsed.category == "technical"
    assert response.parsed.confidence == 0.95
    assert response.metadata.correlation_id == "corr-structured-123"
    assert response.metadata.request_id == "resp_structured_123"
    assert response.metadata.retry_count == 0
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 10
    assert response.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_generate_structured_maps_validation_error() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    async def fake_parse(**kwargs):
        ClassificationResult.model_validate(
            {
                "category": "technical",
                "confidence": None,
            }
        )

    client.client.responses.parse = fake_parse

    with pytest.raises(LLMStructuredValidationError):
        await client.generate_structured(
            "Classify this request.",
            ClassificationResult,
        )


class FakeUnparsedResponse:
    id = "resp_unparsed_123"
    usage = FakeUsage()
    output_parsed = None
    output: ClassVar[list[object]] = []


@pytest.mark.asyncio
async def test_generate_structured_raises_parse_error() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    async def fake_parse(**kwargs):
        return FakeUnparsedResponse()

    client.client.responses.parse = fake_parse

    with pytest.raises(LLMStructuredParseError):
        await client.generate_structured(
            "Classify this request.",
            ClassificationResult,
        )


class FakeRefusalContent:
    type = "refusal"
    refusal = "I cannot comply with this request."


class FakeRefusalMessage:
    type = "message"
    content: ClassVar[list[object]] = [FakeRefusalContent()]


class FakeRefusalResponse:
    id = "resp_refusal_123"
    usage = FakeUsage()
    output_parsed = None
    output: ClassVar[list[object]] = [FakeRefusalMessage()]


@pytest.mark.asyncio
async def test_generate_structured_raises_refusal_error() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    async def fake_parse(**kwargs):
        return FakeRefusalResponse()

    client.client.responses.parse = fake_parse

    with pytest.raises(LLMRefusalError):
        await client.generate_structured(
            "Classify this request.",
            ClassificationResult,
        )


@pytest.mark.asyncio
async def test_generate_structured_from_template_propagates_prompt_identity() -> None:
    from agent_platform.llm.prompt import PromptTemplate

    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    captured_input: str | None = None

    async def fake_parse(
        *,
        model: str,
        input: str,
        text_format: type[BaseModel],
    ) -> FakeStructuredResponse:
        nonlocal captured_input
        captured_input = input
        return FakeStructuredResponse()

    client.client.responses.parse = fake_parse

    prompt = PromptTemplate(
        name="ticket_classifier",
        version="2.0",
        template="Classify this ticket: {ticket_text}",
    )

    response = await client.generate_structured_from_template(
        prompt,
        ClassificationResult,
        {
            "ticket_text": "My card was charged twice.",
        },
        correlation_id="corr-structured-template-123",
    )

    assert captured_input == ("Classify this ticket: My card was charged twice.")
    assert response.metadata.prompt_name == "ticket_classifier"
    assert response.metadata.prompt_version == "2.0"
    assert response.metadata.correlation_id == "corr-structured-template-123"


@pytest.mark.asyncio
async def test_structured_template_fails_before_provider_call() -> None:
    from agent_platform.llm.errors import LLMPromptVariableError
    from agent_platform.llm.prompt import PromptTemplate

    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    calls = 0

    async def fake_parse(
        *,
        model: str,
        input: str,
        text_format: type[BaseModel],
    ) -> FakeStructuredResponse:
        nonlocal calls
        calls += 1
        return FakeStructuredResponse()

    client.client.responses.parse = fake_parse

    prompt = PromptTemplate(
        name="ticket_classifier",
        version="2.0",
        template="Classify this ticket: {ticket_text}",
    )

    with pytest.raises(LLMPromptVariableError):
        await client.generate_structured_from_template(
            prompt,
            ClassificationResult,
            {},
        )

    assert calls == 0


@pytest.mark.asyncio
async def test_generic_structured_prompt_has_no_prompt_identity() -> None:
    client = OpenAIClient(
        Settings(
            openai_api_key="test-key",
        )
    )

    async def fake_parse(
        *,
        model: str,
        input: str,
        text_format: type[BaseModel],
    ) -> FakeStructuredResponse:
        return FakeStructuredResponse()

    client.client.responses.parse = fake_parse

    response = await client.generate_structured(
        "Classify this request.",
        ClassificationResult,
    )

    assert response.metadata.prompt_name is None
    assert response.metadata.prompt_version is None
