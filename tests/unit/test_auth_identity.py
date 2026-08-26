from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
    IdentityType,
)


def test_authenticated_identity_preserves_values() -> None:
    identity = AuthenticatedIdentity(
        subject="user-123",
        identity_type=IdentityType.USER,
        roles=frozenset(
            {
                "llm-user",
                "admin",
            }
        ),
        scopes=frozenset(
            {
                "llm:generate",
                "llm:health",
            }
        ),
    )

    assert identity.subject == "user-123"
    assert identity.identity_type is IdentityType.USER
    assert identity.authenticated is True

    assert identity.roles == frozenset(
        {
            "llm-user",
            "admin",
        }
    )

    assert identity.scopes == frozenset(
        {
            "llm:generate",
            "llm:health",
        }
    )


def test_anonymous_identity_is_unauthenticated() -> None:
    identity = AuthenticatedIdentity.anonymous()

    assert identity.subject == "anonymous"

    assert identity.identity_type is IdentityType.ANONYMOUS

    assert identity.authenticated is False
    assert identity.roles == frozenset()
    assert identity.scopes == frozenset()


def test_identity_has_role() -> None:
    identity = AuthenticatedIdentity(
        subject="user-123",
        identity_type=IdentityType.USER,
        roles=frozenset(
            {
                "llm-user",
            }
        ),
    )

    assert identity.has_role("llm-user")

    assert not identity.has_role("admin")


def test_identity_has_scope() -> None:
    identity = AuthenticatedIdentity(
        subject="service-123",
        identity_type=IdentityType.SERVICE,
        scopes=frozenset(
            {
                "llm:generate",
            }
        ),
    )

    assert identity.has_scope("llm:generate")

    assert not identity.has_scope("llm:admin")
