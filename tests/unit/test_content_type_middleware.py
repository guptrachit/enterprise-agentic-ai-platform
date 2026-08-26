from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.security.content_type_middleware import (
    JSONContentTypeMiddleware,
)


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(JSONContentTypeMiddleware)

    @app.post("/llm/generate")
    async def generate() -> dict[str, str]:
        return {
            "status": "ok",
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "healthy",
        }

    return app


def test_json_content_type_is_allowed() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 200


def test_text_plain_is_rejected_before_route() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        content='{"prompt":"Answer."}',
        headers={
            "Content-Type": "text/plain",
            "X-Correlation-ID": "corr-content-type",
        },
    )

    assert response.status_code == 415

    assert response.json()["detail"] == {
        "code": "unsupported_media_type",
        "message": "Content-Type must be application/json.",
    }

    assert response.headers["X-Correlation-ID"] == "corr-content-type"


def test_missing_content_type_is_rejected() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/llm/generate",
        content='{"prompt":"Answer."}',
    )

    assert response.status_code == 415


def test_unrelated_health_route_is_not_restricted() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
