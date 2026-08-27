import asyncio
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from agent_platform.config import Settings
from agent_platform.llm.api_guardrails import (
    InFlightRequestGuard,
)
from agent_platform.llm.api_models import (
    LLMGenerateResponse,
)
from agent_platform.llm.api_rate_limit import (
    LLMAPIRateLimiter,
)
from agent_platform.llm.api_router import (
    get_api_rate_limiter,
    get_governed_llm_api_service,
    get_in_flight_request_guard,
)
from agent_platform.main import app
from agent_platform.security.auth_dependency import (
    get_authenticated_identity,
)
from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
    IdentityType,
)
from agent_platform.security.scopes import (
    SecurityScope,
)


async def authorized_identity() -> AuthenticatedIdentity:
    return AuthenticatedIdentity(
        subject="integration-user",
        identity_type=IdentityType.USER,
        scopes=frozenset(
            {
                SecurityScope.LLM_GENERATE.value,
            }
        ),
    )


def clear_overrides() -> None:
    app.dependency_overrides.pop(
        get_governed_llm_api_service,
        None,
    )

    app.dependency_overrides.pop(
        get_authenticated_identity,
        None,
    )

    app.dependency_overrides.pop(
        get_api_rate_limiter,
        None,
    )

    app.dependency_overrides.pop(
        get_in_flight_request_guard,
        None,
    )


def test_timeout_returns_504(
    monkeypatch,
) -> None:
    service = AsyncMock()

    async def slow_generate(_):
        await asyncio.sleep(0.05)

    service.generate.side_effect = slow_generate

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_request_timeout_seconds=0.001,
        ),
    )

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    app.dependency_overrides[get_authenticated_identity] = authorized_identity

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                headers={
                    "X-Correlation-ID": "corr-e2e-timeout",
                },
                json={
                    "prompt": "Timeout.",
                },
            )

        assert response.status_code == 504

        assert response.json()["detail"]["code"] == ("llm_request_timeout")

        assert response.headers["X-Correlation-ID"] == "corr-e2e-timeout"

    finally:
        clear_overrides()


def test_rate_limit_returns_429() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="ok",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    app.dependency_overrides[get_authenticated_identity] = authorized_identity

    app.dependency_overrides[get_api_rate_limiter] = lambda: limiter

    try:
        with TestClient(app) as client:
            first = client.post(
                "/llm/generate",
                json={
                    "prompt": "First.",
                },
            )

            second = client.post(
                "/llm/generate",
                headers={
                    "X-Correlation-ID": "corr-e2e-rate",
                },
                json={
                    "prompt": "Second.",
                },
            )

        assert first.status_code == 200
        assert second.status_code == 429

        assert second.json()["detail"]["code"] == ("llm_rate_limit_exceeded")

        assert second.headers["X-Correlation-ID"] == "corr-e2e-rate"

    finally:
        clear_overrides()


def test_capacity_exhaustion_returns_503() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="ok",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    guard = InFlightRequestGuard(max_in_flight=1)

    guard._active_requests = 1

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    app.dependency_overrides[get_authenticated_identity] = authorized_identity

    app.dependency_overrides[get_in_flight_request_guard] = lambda: guard

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                headers={
                    "X-Correlation-ID": "corr-e2e-capacity",
                },
                json={
                    "prompt": "Capacity.",
                },
            )

        assert response.status_code == 503

        assert response.json()["detail"]["code"] == ("llm_capacity_exceeded")

        assert response.headers["X-Correlation-ID"] == "corr-e2e-capacity"

    finally:
        clear_overrides()


def test_bad_content_type_returns_415() -> None:
    service = AsyncMock()

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                content='{"prompt":"bad"}',
                headers={
                    "Content-Type": "text/plain",
                    "X-Correlation-ID": "corr-e2e-media",
                },
            )

        assert response.status_code == 415

        assert response.json()["detail"]["code"] == ("unsupported_media_type")

        assert response.headers["X-Correlation-ID"] == "corr-e2e-media"

    finally:
        clear_overrides()


def test_missing_scope_returns_403() -> None:
    service = AsyncMock()

    async def identity_without_scope() -> AuthenticatedIdentity:
        return AuthenticatedIdentity(
            subject="integration-user-no-scope",
            identity_type=IdentityType.USER,
            scopes=frozenset(),
        )

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    app.dependency_overrides[get_authenticated_identity] = identity_without_scope

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                headers={
                    "X-Correlation-ID": "corr-e2e-authz",
                },
                json={
                    "prompt": "Denied.",
                },
            )

        assert response.status_code == 403

        assert response.json()["detail"]["code"] == ("authorization_denied")

        assert response.headers["X-Correlation-ID"] == "corr-e2e-authz"

    finally:
        clear_overrides()


def test_failure_responses_keep_security_headers() -> None:
    service = AsyncMock()

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                content='{"prompt":"bad"}',
                headers={
                    "Content-Type": "text/plain",
                },
            )

        assert response.status_code == 415

        assert response.headers["X-Content-Type-Options"] == "nosniff"

        assert response.headers["X-Frame-Options"] == "DENY"

        assert response.headers["Cache-Control"] == "no-store"

    finally:
        clear_overrides()
