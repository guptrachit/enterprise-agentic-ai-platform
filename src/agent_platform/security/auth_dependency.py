from typing import Annotated

from fastapi import Depends, Request

from agent_platform.security.anonymous_auth_resolver import (
    AnonymousAuthenticationResolver,
)
from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
)
from agent_platform.security.auth_resolver import (
    AuthenticationResolver,
)


def get_authentication_resolver() -> AuthenticationResolver:
    """Return the configured authentication resolver."""

    return AnonymousAuthenticationResolver()


AuthenticationResolverDependency = Annotated[
    AuthenticationResolver,
    Depends(get_authentication_resolver),
]


async def get_authenticated_identity(
    request: Request,
    resolver: AuthenticationResolverDependency,
) -> AuthenticatedIdentity:
    """Resolve the normalized identity for one HTTP request."""

    return await resolver.resolve(request)


AuthenticatedIdentityDependency = Annotated[
    AuthenticatedIdentity,
    Depends(get_authenticated_identity),
]
