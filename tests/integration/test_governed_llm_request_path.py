from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from agent_platform.llm.api_models import (
    LLMGenerateResponse,
)
from agent_platform.llm.api_router import (
    get_governed_llm_api_service,
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


def test_governed_llm_request_path_end_to_end() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Governed integration response",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    async def authenticated_identity() -> AuthenticatedIdentity:
        return AuthenticatedIdentity(
            subject="integration-user",
            identity_type=IdentityType.USER,
            scopes=frozenset(
                {
                    SecurityScope.LLM_GENERATE.value,
                }
            ),
        )

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    app.dependency_overrides[get_authenticated_identity] = authenticated_identity

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                headers={
                    "X-Correlation-ID": "corr-e2e-001",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": "Explain governed AI.",
                },
            )

        assert response.status_code == 200

        assert response.json() == {
            "content": "Governed integration response",
            "policy_identifier": ("production-routing-policy@1.0.0"),
            "model": "primary",
            "provider": "openai",
        }

        assert response.headers["X-Correlation-ID"] == "corr-e2e-001"

        assert response.headers["X-Content-Type-Options"] == "nosniff"

        assert response.headers["X-Frame-Options"] == "DENY"

        assert response.headers["Cache-Control"] == "no-store"

        service.generate.assert_awaited_once()

        request = service.generate.await_args.args[0]

        assert request.prompt == "Explain governed AI."

        assert request.correlation_id == "corr-e2e-001"

    finally:
        app.dependency_overrides.pop(
            get_governed_llm_api_service,
            None,
        )

        app.dependency_overrides.pop(
            get_authenticated_identity,
            None,
        )


def test_governed_llm_request_path_rejects_missing_scope() -> None:
    service = AsyncMock()

    async def authenticated_identity() -> AuthenticatedIdentity:
        return AuthenticatedIdentity(
            subject="integration-user-no-scope",
            identity_type=IdentityType.USER,
            scopes=frozenset(),
        )

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    app.dependency_overrides[get_authenticated_identity] = authenticated_identity

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                headers={
                    "X-Correlation-ID": "corr-e2e-authz",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": "This should be rejected.",
                },
            )

        assert response.status_code == 403

        assert response.json()["detail"] == {
            "code": "authorization_denied",
            "message": (
                "The authenticated identity is not authorized "
                "to perform this operation."
            ),
        }

        assert response.headers["X-Correlation-ID"] == "corr-e2e-authz"

        service.generate.assert_not_awaited()

    finally:
        app.dependency_overrides.pop(
            get_governed_llm_api_service,
            None,
        )

        app.dependency_overrides.pop(
            get_authenticated_identity,
            None,
        )


def test_governed_llm_request_path_rejects_bad_content_type() -> None:
    service = AsyncMock()

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/llm/generate",
                headers={
                    "X-Correlation-ID": "corr-e2e-media",
                    "Content-Type": "text/plain",
                },
                content='{"prompt":"Rejected."}',
            )

        assert response.status_code == 415

        assert response.json()["detail"]["code"] == ("unsupported_media_type")

        assert response.headers["X-Correlation-ID"] == "corr-e2e-media"

        service.generate.assert_not_awaited()

    finally:
        app.dependency_overrides.pop(
            get_governed_llm_api_service,
            None,
        )
