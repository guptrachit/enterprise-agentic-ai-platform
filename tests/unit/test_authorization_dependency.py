import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.security.auth_dependency import (
    get_authenticated_identity,
)
from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
    IdentityType,
)
from agent_platform.security.authorization import (
    AuthorizationDeniedError,
)
from agent_platform.security.authorization_dependency import (
    LLMGenerateIdentityDependency,
)
from agent_platform.security.scopes import (
    SecurityScope,
)


def test_authorization_dependency_allows_required_scope() -> None:
    app = FastAPI()

    async def authenticated_identity() -> AuthenticatedIdentity:
        return AuthenticatedIdentity(
            subject="user-123",
            identity_type=IdentityType.USER,
            scopes=frozenset(
                {
                    SecurityScope.LLM_GENERATE.value,
                }
            ),
        )

    app.dependency_overrides[get_authenticated_identity] = authenticated_identity

    @app.get("/protected")
    async def protected(
        identity: LLMGenerateIdentityDependency,
    ):
        return {
            "subject": identity.subject,
        }

    client = TestClient(app)

    response = client.get("/protected")

    assert response.status_code == 200

    assert response.json() == {
        "subject": "user-123",
    }


def test_authorization_dependency_raises_for_missing_scope() -> None:
    app = FastAPI()

    async def authenticated_identity() -> AuthenticatedIdentity:
        return AuthenticatedIdentity(
            subject="user-123",
            identity_type=IdentityType.USER,
            scopes=frozenset(),
        )

    app.dependency_overrides[get_authenticated_identity] = authenticated_identity

    @app.get("/protected")
    async def protected(
        identity: LLMGenerateIdentityDependency,
    ):
        return {
            "subject": identity.subject,
        }

    client = TestClient(
        app,
        raise_server_exceptions=True,
    )

    with pytest.raises(
        AuthorizationDeniedError,
    ):
        client.get("/protected")


def test_authorization_dependency_rejects_wrong_scope() -> None:
    app = FastAPI()

    async def authenticated_identity() -> AuthenticatedIdentity:
        return AuthenticatedIdentity(
            subject="user-456",
            identity_type=IdentityType.USER,
            scopes=frozenset(
                {
                    SecurityScope.LLM_HEALTH.value,
                }
            ),
        )

    app.dependency_overrides[get_authenticated_identity] = authenticated_identity

    @app.get("/protected")
    async def protected(
        identity: LLMGenerateIdentityDependency,
    ):
        return {
            "subject": identity.subject,
        }

    client = TestClient(
        app,
        raise_server_exceptions=True,
    )

    with pytest.raises(
        AuthorizationDeniedError,
    ):
        client.get("/protected")
