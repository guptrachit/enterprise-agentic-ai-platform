from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://app.example.com",
        ],
        allow_credentials=False,
        allow_methods=[
            "GET",
            "POST",
            "OPTIONS",
        ],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Correlation-ID",
        ],
    )

    @app.get("/test")
    def test_endpoint() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return app


def test_cors_allows_configured_origin() -> None:
    client = TestClient(create_app())

    response = client.get(
        "/test",
        headers={
            "Origin": "https://app.example.com",
        },
    )

    assert response.status_code == 200

    assert response.headers["Access-Control-Allow-Origin"] == "https://app.example.com"


def test_cors_does_not_allow_unknown_origin() -> None:
    client = TestClient(create_app())

    response = client.get(
        "/test",
        headers={
            "Origin": "https://evil.example.com",
        },
    )

    assert response.status_code == 200

    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_preflight_allows_post() -> None:
    client = TestClient(create_app())

    response = client.options(
        "/test",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": ("Content-Type,X-Correlation-ID"),
        },
    )

    assert response.status_code == 200

    assert response.headers["Access-Control-Allow-Origin"] == "https://app.example.com"

    assert "POST" in response.headers["Access-Control-Allow-Methods"]


def test_cors_preflight_rejects_unknown_origin() -> None:
    client = TestClient(create_app())

    response = client.options(
        "/test",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
