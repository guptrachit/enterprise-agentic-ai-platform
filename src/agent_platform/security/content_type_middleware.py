from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from agent_platform.correlation import (
    CORRELATION_ID_HEADER,
    create_correlation_id,
)
from agent_platform.security.content_type_policy import (
    JSONContentTypePolicy,
    UnsupportedMediaTypeError,
)


class JSONContentTypeMiddleware(BaseHTTPMiddleware):
    """Enforce JSON media type before request body parsing."""

    def __init__(
        self,
        app: ASGIApp,
    ) -> None:
        super().__init__(app)

        self._policy = JSONContentTypePolicy()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[
            [Request],
            Awaitable[Response],
        ],
    ) -> Response:
        """Reject unsupported media types before FastAPI validation."""

        if request.method == "POST" and request.url.path == "/llm/generate":
            try:
                self._policy.enforce(request)
            except UnsupportedMediaTypeError:
                correlation_id = (
                    request.headers.get(CORRELATION_ID_HEADER)
                    or create_correlation_id()
                )

                return JSONResponse(
                    status_code=415,
                    content={
                        "detail": {
                            "code": "unsupported_media_type",
                            "message": ("Content-Type must be application/json."),
                        }
                    },
                    headers={
                        CORRELATION_ID_HEADER: correlation_id,
                    },
                )

        return await call_next(request)
