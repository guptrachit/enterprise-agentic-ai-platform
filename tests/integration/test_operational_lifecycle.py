from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from agent_platform.main import app


def test_operational_lifecycle_end_to_end(
    monkeypatch,
) -> None:
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

    with TestClient(app) as client:
        live_response = client.get("/health/live")

        ready_response = client.get("/health/ready")

        llm_health_response = client.get("/health/llm")

        assert live_response.status_code == 200

        assert live_response.json() == {
            "status": "alive",
        }

        assert ready_response.status_code == 200

        assert ready_response.json() == {
            "status": "ready",
            "reasons": [],
        }

        assert llm_health_response.status_code == 200

        payload = llm_health_response.json()

        assert payload["health"]["status"] in {
            "healthy",
            "degraded",
            "saturated",
        }

        assert payload["runtime"]["resolved"] is True

        assert "api" in payload
        assert "concurrency" in payload
        assert "rate_limit" in payload
        assert "security" in payload
        assert "export" in payload

    refresh_service.close.assert_awaited_once_with()

    runtime.close.assert_awaited_once_with()


def test_readiness_returns_503_when_runtime_becomes_unresolved(
    monkeypatch,
) -> None:
    runtime = Mock()
    runtime.close = AsyncMock()

    refresh_service = Mock()
    refresh_service.close = AsyncMock()

    startup_snapshot = Mock()

    startup_snapshot.to_dict.return_value = {
        "resolved": True,
        "policy_name": "production-routing-policy",
        "policy_identifier": ("production-routing-policy@1.0.0"),
    }

    refresh_service.health_snapshot.return_value = startup_snapshot

    runtime.refresh_service = refresh_service

    monkeypatch.setattr(
        "agent_platform.main.create_fully_configured_governed_runtime",
        lambda **_: runtime,
    )

    with TestClient(app) as client:
        unresolved_snapshot = Mock()

        unresolved_snapshot.to_dict.return_value = {
            "resolved": False,
        }

        refresh_service.health_snapshot.return_value = unresolved_snapshot

        response = client.get("/health/ready")

        assert response.status_code == 503

        assert response.json() == {
            "status": "not_ready",
            "reasons": [
                "routing_policy_unresolved",
            ],
        }
