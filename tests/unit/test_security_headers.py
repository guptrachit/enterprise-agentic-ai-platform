from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.security.security_headers import (
    SECURITY_RESPONSE_HEADERS,
    SecurityHeadersMiddleware,
)


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/test")
    def test_endpoint() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return app


def test_security_headers_are_added() -> None:
    client = TestClient(create_app())

    response = client.get("/test")

    assert response.status_code == 200

    for header, value in SECURITY_RESPONSE_HEADERS.items():
        assert response.headers[header] == value


def test_content_type_options_is_nosniff() -> None:
    client = TestClient(create_app())

    response = client.get("/test")

    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_frame_options_is_deny() -> None:
    client = TestClient(create_app())

    response = client.get("/test")

    assert response.headers["X-Frame-Options"] == "DENY"


def test_referrer_policy_is_no_referrer() -> None:
    client = TestClient(create_app())

    response = client.get("/test")

    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_cache_control_is_no_store() -> None:
    client = TestClient(create_app())

    response = client.get("/test")

    assert response.headers["Cache-Control"] == "no-store"
