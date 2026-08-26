from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends

from agent_platform.security.auth_dependency import (
    AuthenticatedIdentityDependency,
)
from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
)
from agent_platform.security.authorization import (
    ScopeAuthorizationPolicy,
)
from agent_platform.security.scopes import (
    SecurityScope,
)

AuthorizationDependency = Callable[
    [AuthenticatedIdentity],
    Coroutine[Any, Any, AuthenticatedIdentity],
]


def require_scope(
    required_scope: SecurityScope,
) -> AuthorizationDependency:
    """Create a dependency that enforces one authorization scope."""

    async def authorize(
        identity: AuthenticatedIdentityDependency,
    ) -> AuthenticatedIdentity:
        ScopeAuthorizationPolicy(required_scope=required_scope).enforce(identity)

        return identity

    return authorize


LLMGenerateIdentityDependency = Annotated[
    AuthenticatedIdentity,
    Depends(require_scope(SecurityScope.LLM_GENERATE)),
]
