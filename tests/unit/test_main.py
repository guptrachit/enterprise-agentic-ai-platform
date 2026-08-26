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

    assert payload["status"] in {
        "healthy",
        "degraded",
        "saturated",
    }

    assert "runtime" in payload
    assert "api_metrics" in payload
    assert "concurrency" in payload
    assert "rate_limit" in payload

    assert payload["runtime"]["policy_name"] == "production-routing-policy"


def test_llm_health_endpoint_does_not_expose_secrets() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    body = response.text.lower()

    assert "openai_api_key" not in body
    assert "authorization:" not in body
    assert "bearer " not in body
    assert "bearer " not in body
    assert "api-key" not in body

    payload = response.json()

    assert "prompt" not in payload
    assert "request_body" not in payload


def test_llm_health_endpoint_contains_concurrency_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    concurrency = response.json()["concurrency"]

    assert concurrency["max_in_flight"] > 0

    assert concurrency["active_requests"] >= 0

    assert concurrency["available_capacity"] >= 0

    assert concurrency["capacity_rejections"] >= 0

    assert 0.0 <= concurrency["utilization_rate"] <= 1.0


def test_llm_health_endpoint_contains_rate_limit_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    rate_limit = response.json()["rate_limit"]

    assert rate_limit["max_requests"] > 0

    assert rate_limit["window_seconds"] > 0

    assert rate_limit["tracked_callers"] >= 0

    assert rate_limit["rejected_requests"] >= 0


def test_llm_health_endpoint_contains_reason_codes() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    payload = response.json()

    assert "reasons" in payload
    assert isinstance(
        payload["reasons"],
        list,
    )

    assert all(
        isinstance(
            reason,
            str,
        )
        for reason in payload["reasons"]
    )


def test_llm_health_endpoint_contains_security_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    security = response.json()["security"]

    assert security["authentication_failures"] >= 0

    assert security["authorization_failures"] >= 0

    assert security["total_security_failures"] >= 0

    assert isinstance(
        security["failure_counts"],
        dict,
    )


def test_application_adds_security_headers() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert response.headers["X-Frame-Options"] == "DENY"

    assert response.headers["Referrer-Policy"] == "no-referrer"

    assert response.headers["Cache-Control"] == "no-store"


def test_application_allows_configured_cors_origin() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
            },
        )

    assert response.status_code == 200

    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"


def test_application_rejects_unsupported_media_type_with_security_headers() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/llm/generate",
            content='{"prompt":"Answer."}',
            headers={
                "Content-Type": "text/plain",
                "X-Correlation-ID": "corr-main-media",
            },
        )

    assert response.status_code == 415

    assert response.json()["detail"]["code"] == ("unsupported_media_type")

    assert response.headers["X-Correlation-ID"] == "corr-main-media"

    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert response.headers["Cache-Control"] == "no-store"
