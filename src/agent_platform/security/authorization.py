from dataclasses import dataclass

from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
)
from agent_platform.security.scopes import (
    SecurityScope,
)


class AuthorizationDeniedError(PermissionError):
    """Raised when an identity lacks a required permission."""


@dataclass(frozen=True)
class ScopeAuthorizationPolicy:
    """Authorize identities using centralized platform scopes."""

    required_scope: SecurityScope

    def enforce(
        self,
        identity: AuthenticatedIdentity,
    ) -> None:
        """Require the configured scope for an authenticated identity."""

        if not identity.authenticated:
            return

        if not identity.has_scope(self.required_scope.value):
            raise AuthorizationDeniedError(
                f"Required scope is missing: {self.required_scope.value}"
            )
