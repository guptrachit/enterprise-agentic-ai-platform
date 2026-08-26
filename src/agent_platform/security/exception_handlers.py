from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent_platform.correlation import (
    CORRELATION_ID_HEADER,
    create_correlation_id,
)
from agent_platform.security.auth_policy import (
    AuthenticationRequiredError,
)
from agent_platform.security.authorization import (
    AuthorizationDeniedError,
)
from agent_platform.security.security_metrics import (
    SecurityMetrics,
)
from agent_platform.security.security_telemetry import (
    SecurityEventType,
    create_security_event,
    log_security_event,
)

security_metrics = SecurityMetrics()


def _correlation_id_from_request(
    request: Request,
) -> str:
    """Return an existing or generated correlation identifier."""

    return request.headers.get(CORRELATION_ID_HEADER) or create_correlation_id()


async def authentication_required_handler(
    request: Request,
    error: AuthenticationRequiredError,
) -> JSONResponse:
    """Map dependency-time authentication failures."""

    del error

    correlation_id = _correlation_id_from_request(request)

    security_metrics.record_authentication_failure(
        failure_code="authentication_required"
    )

    log_security_event(
        create_security_event(
            correlation_id=correlation_id,
            event_type=(SecurityEventType.AUTHENTICATION_FAILURE),
            status_code=401,
            failure_code="authentication_required",
            path=request.url.path,
        )
    )

    return JSONResponse(
        status_code=401,
        content={
            "detail": {
                "code": "authentication_required",
                "message": "Authentication is required.",
            }
        },
        headers={
            CORRELATION_ID_HEADER: correlation_id,
        },
    )


async def authorization_denied_handler(
    request: Request,
    error: AuthorizationDeniedError,
) -> JSONResponse:
    """Map dependency-time authorization failures."""

    del error

    correlation_id = _correlation_id_from_request(request)

    security_metrics.record_authorization_failure(failure_code="authorization_denied")

    log_security_event(
        create_security_event(
            correlation_id=correlation_id,
            event_type=(SecurityEventType.AUTHORIZATION_FAILURE),
            status_code=403,
            failure_code="authorization_denied",
            path=request.url.path,
        )
    )

    return JSONResponse(
        status_code=403,
        content={
            "detail": {
                "code": "authorization_denied",
                "message": (
                    "The authenticated identity is not authorized "
                    "to perform this operation."
                ),
            }
        },
        headers={
            CORRELATION_ID_HEADER: correlation_id,
        },
    )


def register_security_exception_handlers(
    app: FastAPI,
) -> None:
    """Register global security exception handlers."""

    app.add_exception_handler(
        AuthenticationRequiredError,
        authentication_required_handler,
    )

    app.add_exception_handler(
        AuthorizationDeniedError,
        authorization_denied_handler,
    )
