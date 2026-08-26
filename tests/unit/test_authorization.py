import pytest

from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
    IdentityType,
)
from agent_platform.security.authorization import (
    AuthorizationDeniedError,
    ScopeAuthorizationPolicy,
)
from agent_platform.security.scopes import (
    SecurityScope,
)


def test_scope_policy_allows_required_scope() -> None:
    identity = AuthenticatedIdentity(
        subject="user-123",
        identity_type=IdentityType.USER,
        scopes=frozenset(
            {
                SecurityScope.LLM_GENERATE.value,
            }
        ),
    )

    policy = ScopeAuthorizationPolicy(required_scope=SecurityScope.LLM_GENERATE)

    policy.enforce(identity)


def test_scope_policy_rejects_missing_scope() -> None:
    identity = AuthenticatedIdentity(
        subject="user-123",
        identity_type=IdentityType.USER,
        scopes=frozenset(),
    )

    policy = ScopeAuthorizationPolicy(required_scope=SecurityScope.LLM_GENERATE)

    with pytest.raises(
        AuthorizationDeniedError,
        match="Required scope is missing",
    ):
        policy.enforce(identity)


def test_scope_policy_rejects_wrong_scope() -> None:
    identity = AuthenticatedIdentity(
        subject="service-123",
        identity_type=IdentityType.SERVICE,
        scopes=frozenset(
            {
                SecurityScope.LLM_HEALTH.value,
            }
        ),
    )

    policy = ScopeAuthorizationPolicy(required_scope=SecurityScope.LLM_GENERATE)

    with pytest.raises(AuthorizationDeniedError):
        policy.enforce(identity)


def test_scope_policy_allows_anonymous_for_authentication_layer() -> None:
    identity = AuthenticatedIdentity.anonymous()

    policy = ScopeAuthorizationPolicy(required_scope=SecurityScope.LLM_GENERATE)

    policy.enforce(identity)


def test_scope_matching_is_exact() -> None:
    identity = AuthenticatedIdentity(
        subject="user-123",
        identity_type=IdentityType.USER,
        scopes=frozenset(
            {
                "llm:generate:extended",
            }
        ),
    )

    policy = ScopeAuthorizationPolicy(required_scope=SecurityScope.LLM_GENERATE)

    with pytest.raises(AuthorizationDeniedError):
        policy.enforce(identity)
