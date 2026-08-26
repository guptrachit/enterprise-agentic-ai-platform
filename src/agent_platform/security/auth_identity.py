from dataclasses import dataclass
from enum import StrEnum


class IdentityType(StrEnum):
    """Supported normalized authentication identity types."""

    ANONYMOUS = "anonymous"
    USER = "user"
    SERVICE = "service"
    API_KEY = "api_key"


@dataclass(frozen=True)
class AuthenticatedIdentity:
    """Normalized identity available to application authorization logic."""

    subject: str
    identity_type: IdentityType
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    authenticated: bool = True

    @classmethod
    def anonymous(
        cls,
    ) -> "AuthenticatedIdentity":
        """Create an anonymous unauthenticated identity."""

        return cls(
            subject="anonymous",
            identity_type=IdentityType.ANONYMOUS,
            authenticated=False,
        )

    def has_role(
        self,
        role: str,
    ) -> bool:
        """Return whether the identity has the requested role."""

        return role in self.roles

    def has_scope(
        self,
        scope: str,
    ) -> bool:
        """Return whether the identity has the requested scope."""

        return scope in self.scopes
