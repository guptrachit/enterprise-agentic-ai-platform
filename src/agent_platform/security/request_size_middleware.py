from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from agent_platform.correlation import (
    CORRELATION_ID_HEADER,
    create_correlation_id,
)


class RequestBodyTooLargeError(ValueError):
    """Raised when a request body exceeds the configured maximum."""


class RequestBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized LLM request bodies before application parsing."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be greater than 0")

        super().__init__(app)

        self._max_body_bytes = max_body_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[
            [Request],
            Awaitable[Response],
        ],
    ) -> Response:
        """Reject oversized request bodies."""

        if request.method == "POST" and request.url.path == "/llm/generate":
            content_length = request.headers.get("Content-Length")

            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    declared_length = -1

                if declared_length > self._max_body_bytes:
                    return self._too_large_response(request)

            body = await request.body()

            if len(body) > self._max_body_bytes:
                return self._too_large_response(request)

        return await call_next(request)

    def _too_large_response(
        self,
        request: Request,
    ) -> JSONResponse:
        correlation_id = (
            request.headers.get(CORRELATION_ID_HEADER) or create_correlation_id()
        )

        return JSONResponse(
            status_code=413,
            content={
                "detail": {
                    "code": "request_body_too_large",
                    "message": ("Request body exceeds the configured maximum size."),
                }
            },
            headers={
                CORRELATION_ID_HEADER: correlation_id,
            },
        )
