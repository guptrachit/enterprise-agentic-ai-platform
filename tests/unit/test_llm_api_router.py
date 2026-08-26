from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.llm.api_models import (
    LLMGenerateResponse,
)
from agent_platform.llm.api_router import (
    get_governed_llm_api_service,
    router,
)
from agent_platform.llm.errors import (
    LLMInvalidRequestError,
    LLMTransientError,
)


def create_app(
    service,
) -> FastAPI:
    app = FastAPI()

    app.include_router(router)

    app.dependency_overrides[get_governed_llm_api_service] = lambda: service

    return app


def test_generate_route_returns_governed_response() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Generated answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer this.",
            "correlation_id": "corr-api-001",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "content": "Generated answer",
        "policy_identifier": ("production-routing-policy@1.0.0"),
        "model": "primary",
        "provider": "openai",
    }

    service.generate.assert_awaited_once()

    request = service.generate.await_args.args[0]

    assert request.prompt == "Answer this."
    assert request.correlation_id == "corr-api-001"


def test_generate_route_defaults_workload() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Hello.",
        },
    )

    assert response.status_code == 200

    request = service.generate.await_args.args[0]

    assert request.workload.value == "general"


def test_generate_route_rejects_empty_prompt() -> None:
    service = AsyncMock()

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "",
        },
    )

    assert response.status_code == 422

    service.generate.assert_not_awaited()


def test_generate_route_rejects_invalid_workload() -> None:
    service = AsyncMock()

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
            "workload": "not-a-real-workload",
        },
    )

    assert response.status_code == 422

    service.generate.assert_not_awaited()


def test_unconfigured_dependency_fails_explicitly() -> None:
    app = FastAPI()

    app.include_router(router)

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 500


def test_generate_route_maps_invalid_request_error() -> None:
    service = AsyncMock()

    service.generate.side_effect = LLMInvalidRequestError()

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 400

    assert response.json()["detail"]["code"] == ("invalid_llm_request")


def test_generate_route_maps_transient_error() -> None:
    service = AsyncMock()

    service.generate.side_effect = LLMTransientError()

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 503

    detail = response.json()["detail"]

    assert detail["code"] == ("llm_temporarily_unavailable")

    assert detail["message"] == ("LLM service is temporarily unavailable.")


def test_generate_route_maps_missing_policy() -> None:
    service = AsyncMock()

    service.generate.side_effect = LookupError("No active routing policy found.")

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 503

    assert response.json()["detail"]["code"] == ("routing_policy_unavailable")


def test_generate_route_hides_unexpected_internal_error() -> None:
    service = AsyncMock()

    service.generate.side_effect = RuntimeError("sensitive internal detail")

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 500

    detail = response.json()["detail"]

    assert detail["code"] == "llm_internal_error"

    assert detail["message"] == ("An unexpected LLM runtime error occurred.")

    assert "sensitive internal detail" not in (detail["message"])


def test_generate_route_uses_body_correlation_id() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
            "correlation_id": "corr-body-001",
        },
    )

    assert response.status_code == 200

    assert response.headers["X-Correlation-ID"] == "corr-body-001"

    request = service.generate.await_args.args[0]

    assert request.correlation_id == ("corr-body-001")


def test_generate_route_uses_correlation_header() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        headers={
            "X-Correlation-ID": "corr-header-001",
        },
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 200

    assert response.headers["X-Correlation-ID"] == "corr-header-001"

    request = service.generate.await_args.args[0]

    assert request.correlation_id == ("corr-header-001")


def test_generate_route_generates_correlation_id() -> None:
    from uuid import UUID

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 200

    correlation_id = response.headers["X-Correlation-ID"]

    UUID(correlation_id)

    request = service.generate.await_args.args[0]

    assert request.correlation_id == correlation_id


def test_body_correlation_id_takes_precedence_over_header() -> None:
    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        headers={
            "X-Correlation-ID": "corr-header",
        },
        json={
            "prompt": "Answer.",
            "correlation_id": "corr-body",
        },
    )

    assert response.headers["X-Correlation-ID"] == "corr-body"

    request = service.generate.await_args.args[0]

    assert request.correlation_id == "corr-body"


def test_generate_route_logs_success_telemetry(
    caplog,
) -> None:
    import json
    import logging

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        response = client.post(
            "/llm/generate",
            json={
                "prompt": "Answer.",
                "correlation_id": "corr-api-success",
            },
        )

    assert response.status_code == 200

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_api_request ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("llm_api_request "))

    assert payload["correlation_id"] == ("corr-api-success")

    assert payload["status_code"] == 200
    assert payload["success"] is True

    assert payload["policy_identifier"] == ("production-routing-policy@1.0.0")

    assert payload["model"] == "primary"
    assert payload["provider"] == "openai"

    assert payload["latency_ms"] >= 0


def test_generate_route_logs_failure_telemetry(
    caplog,
) -> None:
    import json
    import logging

    service = AsyncMock()

    service.generate.side_effect = LLMTransientError()

    client = TestClient(create_app(service))

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        response = client.post(
            "/llm/generate",
            json={
                "prompt": "Answer.",
                "correlation_id": "corr-api-failure",
            },
        )

    assert response.status_code == 503

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_api_request ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("llm_api_request "))

    assert payload["correlation_id"] == ("corr-api-failure")

    assert payload["status_code"] == 503
    assert payload["success"] is False
    assert payload["policy_identifier"] is None
    assert payload["model"] is None
    assert payload["provider"] is None

    assert payload["latency_ms"] >= 0


def test_generate_route_does_not_log_prompt(
    caplog,
) -> None:
    import logging

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    secret_prompt = "CONFIDENTIAL_PROMPT_CONTENT_12345"

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        response = client.post(
            "/llm/generate",
            json={
                "prompt": secret_prompt,
            },
        )

    assert response.status_code == 200

    api_logs = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("llm_api_request ")
    )

    assert secret_prompt not in api_logs


def test_generate_route_records_success_metric() -> None:
    from agent_platform.llm.api_router import (
        api_metrics,
    )

    before = api_metrics.snapshot()

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 200

    after = api_metrics.snapshot()

    assert after.total_requests == (before.total_requests + 1)

    assert after.successful_requests == (before.successful_requests + 1)


def test_generate_route_records_failure_metric() -> None:
    from agent_platform.llm.api_router import (
        api_metrics,
    )

    before = api_metrics.snapshot()

    service = AsyncMock()

    service.generate.side_effect = LLMTransientError()

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 503

    after = api_metrics.snapshot()

    assert after.total_requests == (before.total_requests + 1)

    assert after.failed_requests == (before.failed_requests + 1)

    assert after.status_counts[503] >= 1
