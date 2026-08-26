from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.security.request_size_middleware import (
    RequestBodySizeMiddleware,
)


def create_app(
    *,
    max_body_bytes: int = 100,
) -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        RequestBodySizeMiddleware,
        max_body_bytes=max_body_bytes,
    )

    @app.post("/llm/generate")
    async def generate() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return app


def test_request_within_limit_is_allowed() -> None:
    client = TestClient(create_app(max_body_bytes=100))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "small",
        },
    )

    assert response.status_code == 200


def test_oversized_request_is_rejected() -> None:
    client = TestClient(create_app(max_body_bytes=50))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "x" * 100,
        },
        headers={
            "X-Correlation-ID": "corr-body-too-large",
        },
    )

    assert response.status_code == 413

    assert response.json()["detail"] == {
        "code": "request_body_too_large",
        "message": ("Request body exceeds the configured maximum size."),
    }

    assert response.headers["X-Correlation-ID"] == "corr-body-too-large"


def test_request_size_middleware_rejects_invalid_limit() -> None:
    app = FastAPI()

    try:
        RequestBodySizeMiddleware(
            app,
            max_body_bytes=0,
        )
    except ValueError as error:
        assert str(error) == ("max_body_bytes must be greater than 0")
    else:
        raise AssertionError("Expected ValueError")
