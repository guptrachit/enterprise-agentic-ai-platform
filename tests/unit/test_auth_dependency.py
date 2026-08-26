from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.security.anonymous_auth_resolver import (
    AnonymousAuthenticationResolver,
)
from agent_platform.security.auth_dependency import (
    AuthenticatedIdentityDependency,
    get_authenticated_identity,
    get_authentication_resolver,
)
from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
    IdentityType,
)


def test_default_authentication_resolver_is_anonymous() -> None:
    resolver = get_authentication_resolver()

    assert isinstance(
        resolver,
        AnonymousAuthenticationResolver,
    )


@pytest.mark.asyncio
async def test_get_authenticated_identity_uses_resolver() -> None:
    request = Mock()

    resolver = Mock()
    resolver.resolve = AsyncMock(
        return_value=AuthenticatedIdentity(
            subject="user-123",
            identity_type=IdentityType.USER,
            roles=frozenset(
                {
                    "llm-user",
                }
            ),
            scopes=frozenset(
                {
                    "llm:generate",
                }
            ),
        )
    )

    identity = await get_authenticated_identity(
        request=request,
        resolver=resolver,
    )

    resolver.resolve.assert_awaited_once_with(request)

    assert identity.subject == "user-123"
    assert identity.authenticated is True


def test_fastapi_auth_dependency_returns_anonymous_identity() -> None:
    app = FastAPI()

    @app.get("/identity")
    async def identity_endpoint(
        identity: AuthenticatedIdentityDependency,
    ):
        return {
            "subject": identity.subject,
            "identity_type": identity.identity_type.value,
            "authenticated": identity.authenticated,
        }

    client = TestClient(app)

    response = client.get("/identity")

    assert response.status_code == 200

    assert response.json() == {
        "subject": "anonymous",
        "identity_type": "anonymous",
        "authenticated": False,
    }


def test_fastapi_auth_dependency_supports_override() -> None:
    app = FastAPI()

    resolver = Mock()

    resolver.resolve = AsyncMock(
        return_value=AuthenticatedIdentity(
            subject="service-001",
            identity_type=IdentityType.SERVICE,
            scopes=frozenset(
                {
                    "llm:generate",
                }
            ),
        )
    )

    app.dependency_overrides[get_authentication_resolver] = lambda: resolver

    @app.get("/identity")
    async def identity_endpoint(
        identity: AuthenticatedIdentityDependency,
    ):
        return {
            "subject": identity.subject,
            "identity_type": identity.identity_type.value,
            "authenticated": identity.authenticated,
        }

    client = TestClient(app)

    response = client.get("/identity")

    assert response.status_code == 200

    assert response.json() == {
        "subject": "service-001",
        "identity_type": "service",
        "authenticated": True,
    }
