import pytest

from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
    IdentityType,
)
from agent_platform.security.auth_policy import (
    AuthenticationPolicy,
    AuthenticationRequiredError,
)


def test_authentication_policy_allows_anonymous_when_optional() -> None:
    policy = AuthenticationPolicy(authentication_required=False)

    policy.enforce(AuthenticatedIdentity.anonymous())


def test_authentication_policy_rejects_anonymous_when_required() -> None:
    policy = AuthenticationPolicy(authentication_required=True)

    with pytest.raises(
        AuthenticationRequiredError,
        match="Authentication is required",
    ):
        policy.enforce(AuthenticatedIdentity.anonymous())


def test_authentication_policy_allows_authenticated_user() -> None:
    policy = AuthenticationPolicy(authentication_required=True)

    policy.enforce(
        AuthenticatedIdentity(
            subject="user-123",
            identity_type=IdentityType.USER,
        )
    )


def test_authentication_policy_allows_authenticated_service() -> None:
    policy = AuthenticationPolicy(authentication_required=True)

    policy.enforce(
        AuthenticatedIdentity(
            subject="service-001",
            identity_type=IdentityType.SERVICE,
        )
    )
