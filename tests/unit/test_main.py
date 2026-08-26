from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from agent_platform.llm.api_models import LLMGenerateResponse
from agent_platform.main import app
from agent_platform.production_readiness import (
    ProductionReadinessError,
)


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

    assert payload["health"]["status"] in {
        "healthy",
        "degraded",
        "saturated",
    }

    assert "runtime" in payload
    assert "api" in payload
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

    assert "reasons" in payload["health"]
    assert isinstance(
        payload["health"]["reasons"],
        list,
    )

    assert all(
        isinstance(
            reason,
            str,
        )
        for reason in payload["health"]["reasons"]
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


def test_llm_health_endpoint_contains_export_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health/llm")

    assert response.status_code == 200

    export = response.json()["export"]

    assert export["total_exports"] >= 0

    assert export["successful_exports"] >= 0

    assert export["failed_exports"] >= 0

    assert 0.0 <= export["success_rate"] <= 1.0

    assert 0.0 <= export["failure_rate"] <= 1.0


def test_application_startup_runs_production_readiness_validation(
    monkeypatch,
) -> None:
    from agent_platform.config import Settings

    monkeypatch.setattr(
        "agent_platform.main.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            app_env="production",
            llm_api_authentication_required=False,
            llm_api_cors_allowed_origins=("https://app.example.com",),
        ),
    )

    try:
        with TestClient(app):
            pass
    except ProductionReadinessError:
        pass
    else:
        raise AssertionError("Expected startup production readiness failure")


def test_application_startup_fails_when_runtime_is_unresolved(
    monkeypatch,
) -> None:
    from unittest.mock import Mock

    runtime = Mock()

    refresh_service = Mock()

    health_snapshot = Mock()

    health_snapshot.to_dict.return_value = {
        "resolved": False,
    }

    refresh_service.health_snapshot.return_value = health_snapshot

    runtime.refresh_service = refresh_service

    monkeypatch.setattr(
        "agent_platform.main.create_fully_configured_governed_runtime",
        lambda **_: runtime,
    )

    with (
        pytest.raises(
            ProductionReadinessError,
            match="Governed LLM runtime is not ready",
        ),
        TestClient(app),
    ):
        pass


def test_application_shutdown_closes_runtime_resources(
    monkeypatch,
) -> None:
    from unittest.mock import AsyncMock, Mock

    runtime = Mock()
    runtime.close = AsyncMock()

    refresh_service = Mock()
    refresh_service.close = AsyncMock()

    health_snapshot = Mock()

    health_snapshot.to_dict.return_value = {
        "resolved": True,
        "policy_name": "production-routing-policy",
        "policy_identifier": ("production-routing-policy@1.0.0"),
    }

    refresh_service.health_snapshot.return_value = health_snapshot

    runtime.refresh_service = refresh_service

    monkeypatch.setattr(
        "agent_platform.main.create_fully_configured_governed_runtime",
        lambda **_: runtime,
    )

    with TestClient(app):
        pass

    refresh_service.close.assert_awaited_once_with()
    runtime.close.assert_awaited_once_with()


def test_application_logs_startup_and_shutdown_events(
    caplog,
) -> None:
    import logging

    with (
        caplog.at_level(
            logging.INFO,
            logger="agent_platform.lifecycle",
        ),
        TestClient(app),
    ):
        pass

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("lifecycle_event ")
    ]

    assert any("application_startup_succeeded" in message for message in messages)

    assert any("application_shutdown_started" in message for message in messages)

    assert any("application_shutdown_completed" in message for message in messages)


def test_liveness_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200

    assert response.json() == {
        "status": "alive",
    }


def test_readiness_endpoint_returns_ready() -> None:
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ready",
        "reasons": [],
    }


def test_readiness_endpoint_returns_503_when_runtime_unresolved(
    monkeypatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    with TestClient(app) as client:
        runtime = Mock()

        snapshot = Mock()

        snapshot.to_dict.return_value = {
            "resolved": False,
        }

        runtime.health_snapshot.return_value = snapshot

        monkeypatch.setattr(
            app.state,
            "governed_llm_runtime",
            SimpleNamespace(
                refresh_service=runtime,
            ),
        )

        response = client.get("/health/ready")

    assert response.status_code == 503

    assert response.json() == {
        "status": "not_ready",
        "reasons": [
            "routing_policy_unresolved",
        ],
    }
