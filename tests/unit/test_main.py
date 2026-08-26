from unittest.mock import Mock

from fastapi.testclient import TestClient

from agent_platform.llm.api_models import LLMGenerateResponse
from agent_platform.main import app


def test_root_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "application": "Enterprise Agentic AI Platform",
        "version": "0.1.0",
        "status": "running",
    }


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_llm_generate_route_is_registered() -> None:
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/llm/generate" in paths
    assert "post" in paths["/llm/generate"]


def test_llm_generate_uses_application_service() -> None:
    service = Mock()

    async def generate(request):
        assert request.prompt == "Hello."

        return LLMGenerateResponse(
            content="Generated answer",
            policy_identifier=("production-routing-policy@1.0.0"),
            model="primary",
            provider="openai",
        )

    service.generate = generate

    with TestClient(app) as client:
        original_service = app.state.governed_llm_api_service

        try:
            app.state.governed_llm_api_service = service

            response = client.post(
                "/llm/generate",
                json={
                    "prompt": "Hello.",
                },
            )
        finally:
            app.state.governed_llm_api_service = original_service

    assert response.status_code == 200

    assert response.json() == {
        "content": "Generated answer",
        "policy_identifier": ("production-routing-policy@1.0.0"),
        "model": "primary",
        "provider": "openai",
    }


def test_llm_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "healthy"

    assert "runtime" in payload
    assert "api_metrics" in payload

    assert payload["runtime"]["policy_name"] == "production-routing-policy"


def test_llm_health_endpoint_does_not_expose_secrets() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    body = response.text.lower()

    assert "openai_api_key" not in body
    assert "authorization" not in body
    assert "prompt" not in body
