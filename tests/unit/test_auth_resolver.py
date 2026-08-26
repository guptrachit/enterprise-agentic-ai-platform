from unittest.mock import Mock

import pytest

from agent_platform.security.anonymous_auth_resolver import (
    AnonymousAuthenticationResolver,
)
from agent_platform.security.auth_identity import (
    IdentityType,
)
from agent_platform.security.auth_resolver import (
    AuthenticationResolver,
)


def test_authentication_resolver_is_abstract() -> None:
    with pytest.raises(
        TypeError,
    ):
        AuthenticationResolver()


@pytest.mark.asyncio
async def test_anonymous_resolver_returns_anonymous_identity() -> None:
    resolver = AnonymousAuthenticationResolver()

    request = Mock()

    identity = await resolver.resolve(request)

    assert identity.subject == "anonymous"

    assert identity.identity_type is IdentityType.ANONYMOUS

    assert identity.authenticated is False
    assert identity.roles == frozenset()
    assert identity.scopes == frozenset()


@pytest.mark.asyncio
async def test_anonymous_resolver_is_stateless() -> None:
    resolver = AnonymousAuthenticationResolver()

    first = await resolver.resolve(Mock())

    second = await resolver.resolve(Mock())

    assert first == second
