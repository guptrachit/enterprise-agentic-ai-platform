import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.security.auth_policy import (
    AuthenticationRequiredError,
)
from agent_platform.security.authorization import (
    AuthorizationDeniedError,
)
from agent_platform.security.exception_handlers import (
    register_security_exception_handlers,
)


def test_authentication_handler_returns_401() -> None:
    app = FastAPI()

    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected():
        raise AuthenticationRequiredError()

    client = TestClient(app)

    response = client.get(
        "/protected",
        headers={
            "X-Correlation-ID": "corr-auth-001",
        },
    )

    assert response.status_code == 401

    assert response.json()["detail"] == {
        "code": "authentication_required",
        "message": "Authentication is required.",
    }

    assert response.headers["X-Correlation-ID"] == "corr-auth-001"


def test_authorization_handler_returns_403() -> None:
    app = FastAPI()

    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected():
        raise AuthorizationDeniedError()

    client = TestClient(app)

    response = client.get(
        "/protected",
        headers={
            "X-Correlation-ID": "corr-authz-001",
        },
    )

    assert response.status_code == 403

    assert response.json()["detail"] == {
        "code": "authorization_denied",
        "message": (
            "The authenticated identity is not authorized to perform this operation."
        ),
    }

    assert response.headers["X-Correlation-ID"] == "corr-authz-001"


def test_security_handler_generates_correlation_id() -> None:
    app = FastAPI()

    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected():
        raise AuthorizationDeniedError()

    client = TestClient(app)

    response = client.get("/protected")

    assert response.status_code == 403

    assert response.headers["X-Correlation-ID"]


def test_authentication_failure_logs_security_event(
    caplog,
) -> None:
    app = FastAPI()

    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected():
        raise AuthenticationRequiredError()

    client = TestClient(app)

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.security",
    ):
        response = client.get(
            "/protected",
            headers={
                "X-Correlation-ID": "corr-auth-log",
            },
        )

    assert response.status_code == 401

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("security_event ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("security_event "))

    assert payload["correlation_id"] == "corr-auth-log"

    assert payload["event_type"] == ("authentication_failure")

    assert payload["failure_code"] == ("authentication_required")


def test_authorization_failure_logs_security_event(
    caplog,
) -> None:
    app = FastAPI()

    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected():
        raise AuthorizationDeniedError()

    client = TestClient(app)

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.security",
    ):
        response = client.get(
            "/protected",
            headers={
                "X-Correlation-ID": "corr-authz-log",
            },
        )

    assert response.status_code == 403

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("security_event ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("security_event "))

    assert payload["correlation_id"] == "corr-authz-log"

    assert payload["event_type"] == ("authorization_failure")

    assert payload["failure_code"] == ("authorization_denied")

    assert payload["path"] == "/protected"


def test_authentication_handler_records_metric() -> None:
    from agent_platform.security.exception_handlers import (
        security_metrics,
    )

    before = security_metrics.snapshot()

    app = FastAPI()

    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected():
        raise AuthenticationRequiredError()

    client = TestClient(app)

    response = client.get("/protected")

    assert response.status_code == 401

    after = security_metrics.snapshot()

    assert after.authentication_failures == (before.authentication_failures + 1)

    before_count = before.failure_counts.get(
        "authentication_required",
        0,
    )

    after_count = after.failure_counts.get(
        "authentication_required",
        0,
    )

    assert after_count == before_count + 1


def test_authorization_handler_records_metric() -> None:
    from agent_platform.security.exception_handlers import (
        security_metrics,
    )

    before = security_metrics.snapshot()

    app = FastAPI()

    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected():
        raise AuthorizationDeniedError()

    client = TestClient(app)

    response = client.get("/protected")

    assert response.status_code == 403

    after = security_metrics.snapshot()

    assert after.authorization_failures == (before.authorization_failures + 1)

    before_count = before.failure_counts.get(
        "authorization_denied",
        0,
    )

    after_count = after.failure_counts.get(
        "authorization_denied",
        0,
    )

    assert after_count == before_count + 1
