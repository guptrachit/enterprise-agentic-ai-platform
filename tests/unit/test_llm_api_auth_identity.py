from unittest.mock import Mock

from agent_platform.llm.api_router import (
    resolve_rate_limit_identity,
)
from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
    IdentityType,
)


def test_authenticated_identity_uses_subject() -> None:
    request = Mock()

    identity = AuthenticatedIdentity(
        subject="user-123",
        identity_type=IdentityType.USER,
    )

    caller_id = resolve_rate_limit_identity(
        request=request,
        identity=identity,
    )

    assert caller_id == "user-123"


def test_authenticated_service_uses_subject() -> None:
    request = Mock()

    identity = AuthenticatedIdentity(
        subject="service-001",
        identity_type=IdentityType.SERVICE,
    )

    caller_id = resolve_rate_limit_identity(
        request=request,
        identity=identity,
    )

    assert caller_id == "service-001"
