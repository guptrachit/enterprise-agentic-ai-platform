from abc import ABC, abstractmethod

from fastapi import Request

from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
)


class AuthenticationResolver(ABC):
    """Resolve normalized application identity from an HTTP request."""

    @abstractmethod
    async def resolve(
        self,
        request: Request,
    ) -> AuthenticatedIdentity:
        """Resolve the authenticated identity for one request."""
