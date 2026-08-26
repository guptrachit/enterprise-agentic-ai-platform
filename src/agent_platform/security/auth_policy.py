from dataclasses import dataclass

from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
)


class AuthenticationRequiredError(PermissionError):
    """Raised when authentication is required but identity is anonymous."""


@dataclass(frozen=True)
class AuthenticationPolicy:
    """Configurable authentication enforcement policy."""

    authentication_required: bool = False

    def enforce(
        self,
        identity: AuthenticatedIdentity,
    ) -> None:
        """Enforce configured authentication requirements."""

        if self.authentication_required and not identity.authenticated:
            raise AuthenticationRequiredError("Authentication is required.")
