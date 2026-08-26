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
from agent_platform.security.exception_handlers import (
    register_security_exception_handlers,
)


def create_app(
    service,
) -> FastAPI:
    app = FastAPI()

    register_security_exception_handlers(app)

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


def test_generate_route_rejects_oversized_prompt(
    monkeypatch,
) -> None:
    from agent_platform.config import Settings

    service = AsyncMock()

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_max_prompt_chars=5,
        ),
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "123456",
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert detail["code"] == "prompt_too_large"

    service.generate.assert_not_awaited()


def test_generate_route_accepts_prompt_at_limit(
    monkeypatch,
) -> None:
    from agent_platform.config import Settings

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_max_prompt_chars=5,
        ),
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "12345",
        },
    )

    assert response.status_code == 200

    service.generate.assert_awaited_once()


def test_generate_route_maps_timeout_to_504(
    monkeypatch,
) -> None:
    import asyncio

    from agent_platform.config import Settings

    service = AsyncMock()

    async def slow_generate(_):
        await asyncio.sleep(0.05)

    service.generate.side_effect = slow_generate

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_request_timeout_seconds=0.001,
        ),
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
            "correlation_id": "corr-timeout",
        },
    )

    assert response.status_code == 504

    detail = response.json()["detail"]

    assert detail["code"] == "llm_request_timeout"

    assert response.headers["X-Correlation-ID"] == "corr-timeout"


def test_generate_route_rejects_when_capacity_exhausted() -> None:
    from agent_platform.llm.api_guardrails import (
        InFlightRequestGuard,
    )
    from agent_platform.llm.api_router import (
        get_in_flight_request_guard,
    )

    service = AsyncMock()

    guard = InFlightRequestGuard(max_in_flight=1)

    app = create_app(service)

    app.dependency_overrides[get_in_flight_request_guard] = lambda: guard

    async def run_test() -> None:
        async with guard.slot():
            client = TestClient(app)

            response = client.post(
                "/llm/generate",
                json={
                    "prompt": "Answer.",
                    "correlation_id": ("corr-capacity"),
                },
            )

            assert response.status_code == 503

            detail = response.json()["detail"]

            assert detail["code"] == ("llm_capacity_exceeded")

            assert response.headers["X-Correlation-ID"] == "corr-capacity"

    import asyncio

    asyncio.run(run_test())

    service.generate.assert_not_awaited()


def test_generate_route_rejects_rate_limit_exceeded() -> None:
    from agent_platform.llm.api_rate_limit import (
        LLMAPIRateLimiter,
    )
    from agent_platform.llm.api_router import (
        get_api_rate_limiter,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    app = create_app(service)

    app.dependency_overrides[get_api_rate_limiter] = lambda: limiter

    client = TestClient(app)

    first = client.post(
        "/llm/generate",
        headers={
            "X-Client-ID": "client-rate-test",
        },
        json={
            "prompt": "First.",
        },
    )

    second = client.post(
        "/llm/generate",
        headers={
            "X-Client-ID": "client-rate-test",
        },
        json={
            "prompt": "Second.",
            "correlation_id": "corr-rate-limit",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 429

    detail = second.json()["detail"]

    assert detail["code"] == ("llm_rate_limit_exceeded")

    assert second.headers["X-Correlation-ID"] == "corr-rate-limit"

    assert service.generate.await_count == 1


def test_rate_limiter_rejection_count_increments() -> None:
    from agent_platform.llm.api_rate_limit import (
        LLMAPIRateLimiter,
    )
    from agent_platform.llm.api_router import (
        get_api_rate_limiter,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    app = create_app(service)

    app.dependency_overrides[get_api_rate_limiter] = lambda: limiter

    client = TestClient(app)

    client.post(
        "/llm/generate",
        headers={
            "X-Client-ID": "client-health-test",
        },
        json={
            "prompt": "First.",
        },
    )

    response = client.post(
        "/llm/generate",
        headers={
            "X-Client-ID": "client-health-test",
        },
        json={
            "prompt": "Second.",
        },
    )

    assert response.status_code == 429

    snapshot = limiter.snapshot()

    assert snapshot.rejected_requests == 1


def test_forwarded_for_does_not_bypass_rate_limit_when_proxy_untrusted(
    monkeypatch,
) -> None:
    from agent_platform.config import Settings
    from agent_platform.llm.api_rate_limit import (
        LLMAPIRateLimiter,
    )
    from agent_platform.llm.api_router import (
        get_api_rate_limiter,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_trusted_proxy_hosts=(),
        ),
    )

    app = create_app(service)

    app.dependency_overrides[get_api_rate_limiter] = lambda: limiter

    client = TestClient(app)

    first = client.post(
        "/llm/generate",
        headers={
            "X-Forwarded-For": "198.51.100.1",
        },
        json={
            "prompt": "First.",
        },
    )

    second = client.post(
        "/llm/generate",
        headers={
            "X-Forwarded-For": "198.51.100.2",
        },
        json={
            "prompt": "Second.",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 429

    assert service.generate.await_count == 1


def test_failure_telemetry_classifies_rate_limit(
    caplog,
) -> None:
    import json
    import logging

    from agent_platform.llm.api_rate_limit import (
        LLMAPIRateLimiter,
    )
    from agent_platform.llm.api_router import (
        get_api_rate_limiter,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    app = create_app(service)

    app.dependency_overrides[get_api_rate_limiter] = lambda: limiter

    client = TestClient(app)

    client.post(
        "/llm/generate",
        json={
            "prompt": "First.",
        },
    )

    caplog.clear()

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        response = client.post(
            "/llm/generate",
            json={
                "prompt": "Second.",
                "correlation_id": "corr-rate-classification",
            },
        )

    assert response.status_code == 429

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_api_request ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("llm_api_request "))

    assert payload["failure_code"] == ("llm_rate_limit_exceeded")


def test_failure_telemetry_classifies_timeout(
    monkeypatch,
    caplog,
) -> None:
    import asyncio
    import json
    import logging

    from agent_platform.config import Settings

    service = AsyncMock()

    async def slow_generate(_):
        await asyncio.sleep(0.05)

    service.generate.side_effect = slow_generate

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_request_timeout_seconds=0.001,
        ),
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
                "correlation_id": "corr-timeout-classification",
            },
        )

    assert response.status_code == 504

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_api_request ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("llm_api_request "))

    assert payload["failure_code"] == ("llm_request_timeout")


def test_api_metrics_classify_rate_limit_failure() -> None:
    from agent_platform.llm.api_rate_limit import (
        LLMAPIRateLimiter,
    )
    from agent_platform.llm.api_router import (
        api_metrics,
        get_api_rate_limiter,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    app = create_app(service)

    app.dependency_overrides[get_api_rate_limiter] = lambda: limiter

    client = TestClient(app)

    before = api_metrics.snapshot()

    client.post(
        "/llm/generate",
        json={
            "prompt": "First.",
        },
    )

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Second.",
        },
    )

    assert response.status_code == 429

    after = api_metrics.snapshot()

    before_count = before.failure_counts.get(
        "llm_rate_limit_exceeded",
        0,
    )

    after_count = after.failure_counts.get(
        "llm_rate_limit_exceeded",
        0,
    )

    assert after_count == before_count + 1


def test_api_metrics_classify_timeout_failure(
    monkeypatch,
) -> None:
    import asyncio

    from agent_platform.config import Settings
    from agent_platform.llm.api_router import (
        api_metrics,
    )

    service = AsyncMock()

    async def slow_generate(_):
        await asyncio.sleep(0.05)

    service.generate.side_effect = slow_generate

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_request_timeout_seconds=0.001,
        ),
    )

    client = TestClient(create_app(service))

    before = api_metrics.snapshot()

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Timeout.",
        },
    )

    assert response.status_code == 504

    after = api_metrics.snapshot()

    before_count = before.failure_counts.get(
        "llm_request_timeout",
        0,
    )

    after_count = after.failure_counts.get(
        "llm_request_timeout",
        0,
    )

    assert after_count == before_count + 1


def test_authenticated_identity_is_used_for_rate_limit() -> None:
    from agent_platform.llm.api_rate_limit import (
        LLMAPIRateLimiter,
    )
    from agent_platform.llm.api_router import (
        get_api_rate_limiter,
    )
    from agent_platform.security.auth_dependency import (
        get_authenticated_identity,
    )
    from agent_platform.security.auth_identity import (
        AuthenticatedIdentity,
        IdentityType,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    app = create_app(service)

    app.dependency_overrides[get_api_rate_limiter] = lambda: limiter

    app.dependency_overrides[get_authenticated_identity] = lambda: (
        AuthenticatedIdentity(
            subject="user-rate-limit-001",
            identity_type=IdentityType.USER,
            scopes=frozenset(
                {
                    "llm:generate",
                }
            ),
        )
    )

    client = TestClient(app)

    first = client.post(
        "/llm/generate",
        json={
            "prompt": "First.",
        },
    )

    second = client.post(
        "/llm/generate",
        json={
            "prompt": "Second.",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 429

    assert service.generate.await_count == 1

    snapshot = limiter.snapshot()

    assert snapshot.tracked_callers == 1


def test_generate_route_rejects_anonymous_when_auth_required(
    monkeypatch,
) -> None:
    from agent_platform.config import Settings

    service = AsyncMock()

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_authentication_required=True,
        ),
    )

    client = TestClient(create_app(service))

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
            "correlation_id": "corr-auth-required",
        },
    )

    assert response.status_code == 401

    detail = response.json()["detail"]

    assert detail["code"] == ("authentication_required")

    assert response.headers["X-Correlation-ID"] == "corr-auth-required"

    service.generate.assert_not_awaited()


def test_generate_route_allows_authenticated_when_auth_required(
    monkeypatch,
) -> None:
    from agent_platform.config import Settings
    from agent_platform.security.auth_dependency import (
        get_authenticated_identity,
    )
    from agent_platform.security.auth_identity import (
        AuthenticatedIdentity,
        IdentityType,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    monkeypatch.setattr(
        "agent_platform.llm.api_router.get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            llm_api_authentication_required=True,
        ),
    )

    app = create_app(service)

    app.dependency_overrides[get_authenticated_identity] = lambda: (
        AuthenticatedIdentity(
            subject="user-123",
            identity_type=IdentityType.USER,
            scopes=frozenset(
                {
                    "llm:generate",
                }
            ),
        )
    )

    client = TestClient(app)

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 200

    service.generate.assert_awaited_once()


def test_generate_route_rejects_authenticated_identity_without_scope() -> None:
    from agent_platform.security.auth_dependency import (
        get_authenticated_identity,
    )
    from agent_platform.security.auth_identity import (
        AuthenticatedIdentity,
        IdentityType,
    )

    service = AsyncMock()

    app = create_app(service)

    app.dependency_overrides[get_authenticated_identity] = lambda: (
        AuthenticatedIdentity(
            subject="user-no-scope",
            identity_type=IdentityType.USER,
            scopes=frozenset(),
        )
    )

    client = TestClient(app)

    response = client.post(
        "/llm/generate",
        headers={
            "X-Correlation-ID": "corr-no-scope",
        },
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 403

    detail = response.json()["detail"]

    assert detail["code"] == ("authorization_denied")

    assert response.headers["X-Correlation-ID"] == "corr-no-scope"

    service.generate.assert_not_awaited()


def test_generate_route_allows_identity_with_generate_scope() -> None:
    from agent_platform.security.auth_dependency import (
        get_authenticated_identity,
    )
    from agent_platform.security.auth_identity import (
        AuthenticatedIdentity,
        IdentityType,
    )

    service = AsyncMock()

    service.generate.return_value = LLMGenerateResponse(
        content="Answer",
        policy_identifier=("production-routing-policy@1.0.0"),
        model="primary",
        provider="openai",
    )

    app = create_app(service)

    app.dependency_overrides[get_authenticated_identity] = lambda: (
        AuthenticatedIdentity(
            subject="user-authorized",
            identity_type=IdentityType.USER,
            scopes=frozenset(
                {
                    "llm:generate",
                }
            ),
        )
    )

    client = TestClient(app)

    response = client.post(
        "/llm/generate",
        json={
            "prompt": "Answer.",
        },
    )

    assert response.status_code == 200

    service.generate.assert_awaited_once()
