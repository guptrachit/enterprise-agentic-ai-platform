from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.security.content_type_middleware import (
    JSONContentTypeMiddleware,
)
from agent_platform.security.exception_handlers import (
    register_security_exception_handlers,
)
from agent_platform.security.request_size_middleware import (
    RequestBodySizeMiddleware,
)
from agent_platform.security.security_headers import (
    SecurityHeadersMiddleware,
)


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(JSONContentTypeMiddleware)

    app.add_middleware(
        RequestBodySizeMiddleware,
        max_body_bytes=100,
    )

    app.add_middleware(SecurityHeadersMiddleware)

    register_security_exception_handlers(app)

    @app.post("/llm/generate")
    async def generate() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return app


def test_valid_json_request_reaches_route() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "small",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
    }


def test_unsupported_media_type_returns_415() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        content='{"prompt":"small"}',
        headers={
            "Content-Type": "text/plain",
            "X-Correlation-ID": "corr-media",
        },
    )

    assert response.status_code == 415

    assert response.json()["detail"] == {
        "code": "unsupported_media_type",
        "message": "Content-Type must be application/json.",
    }

    assert response.headers["X-Correlation-ID"] == "corr-media"


def test_oversized_json_returns_413() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "x" * 200,
        },
        headers={
            "X-Correlation-ID": "corr-large",
        },
    )

    assert response.status_code == 413

    assert response.json()["detail"] == {
        "code": "request_body_too_large",
        "message": ("Request body exceeds the configured maximum size."),
    }

    assert response.headers["X-Correlation-ID"] == "corr-large"


def test_security_headers_are_present_on_success() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "small",
        },
    )

    assert response.status_code == 200

    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert response.headers["X-Frame-Options"] == "DENY"

    assert response.headers["Referrer-Policy"] == "no-referrer"

    assert response.headers["Cache-Control"] == "no-store"


def test_security_headers_are_present_on_415() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        content='{"prompt":"small"}',
        headers={
            "Content-Type": "text/plain",
        },
    )

    assert response.status_code == 415

    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert response.headers["Cache-Control"] == "no-store"


def test_security_headers_are_present_on_413() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "x" * 200,
        },
    )

    assert response.status_code == 413

    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert response.headers["Cache-Control"] == "no-store"
