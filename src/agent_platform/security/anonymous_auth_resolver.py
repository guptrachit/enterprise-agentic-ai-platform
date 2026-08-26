from fastapi import Request

from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
)
from agent_platform.security.auth_resolver import (
    AuthenticationResolver,
)


class AnonymousAuthenticationResolver(AuthenticationResolver):
    """Resolve every request as anonymous."""

    async def resolve(
        self,
        request: Request,
    ) -> AuthenticatedIdentity:
        """Return an anonymous identity."""

        del request

        return AuthenticatedIdentity.anonymous()
