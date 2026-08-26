import pytest
from pydantic import ValidationError

from agent_platform.llm.api_models import (
    LLMGenerateRequest,
    LLMGenerateResponse,
)
from agent_platform.llm.workload import LLMWorkload


def test_generate_request_defaults_to_general_workload() -> None:
    request = LLMGenerateRequest(
        prompt="Hello.",
    )

    assert request.prompt == "Hello."
    assert request.workload is LLMWorkload.GENERAL
    assert request.correlation_id is None


def test_generate_request_preserves_metadata() -> None:
    request = LLMGenerateRequest(
        prompt="Classify.",
        workload=LLMWorkload.CLASSIFICATION,
        correlation_id="corr-001",
        prompt_name="classification",
        prompt_version="1.0",
    )

    assert request.workload is LLMWorkload.CLASSIFICATION
    assert request.correlation_id == "corr-001"
    assert request.prompt_name == "classification"
    assert request.prompt_version == "1.0"


def test_generate_request_rejects_empty_prompt() -> None:
    with pytest.raises(
        ValidationError,
    ):
        LLMGenerateRequest(
            prompt="",
        )


def test_generate_response_preserves_execution_metadata() -> None:
    response = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    assert response.content == "Answer"

    assert response.policy_identifier == ("production-routing-policy@1.0.0")

    assert response.model == "primary"
    assert response.provider == "openai"
