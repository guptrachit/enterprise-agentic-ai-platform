import asyncio
import time
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from agent_platform.config import get_settings
from agent_platform.correlation import (
    CORRELATION_ID_HEADER,
    create_correlation_id,
)
from agent_platform.llm.api_client_identity import (
    resolve_client_identity,
)
from agent_platform.llm.api_errors import (
    LLMRequestTimeoutError,
    map_llm_exception,
)
from agent_platform.llm.api_guardrails import (
    InFlightRequestGuard,
    validate_prompt_length,
)
from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)
from agent_platform.llm.api_models import (
    LLMGenerateRequest,
    LLMGenerateResponse,
)
from agent_platform.llm.api_rate_limit import (
    LLMAPIRateLimiter,
)
from agent_platform.llm.api_service import (
    GovernedLLMAPIService,
)
from agent_platform.llm.api_telemetry import (
    create_llm_api_request_event,
    log_llm_api_request_event,
)
from agent_platform.security.auth_identity import (
    AuthenticatedIdentity,
)
from agent_platform.security.auth_policy import (
    AuthenticationPolicy,
)
from agent_platform.security.authorization_dependency import (
    LLMGenerateIdentityDependency,
)

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)

api_metrics = LLMAPIMetrics()


@lru_cache
def create_in_flight_request_guard(
    max_in_flight: int,
) -> InFlightRequestGuard:
    """Return a shared concurrency guard for one configured limit."""

    return InFlightRequestGuard(max_in_flight=max_in_flight)


def get_in_flight_request_guard() -> InFlightRequestGuard:
    """Return the configured governed LLM API concurrency guard."""

    settings = get_settings()

    return create_in_flight_request_guard(settings.llm_api_max_in_flight_requests)


@lru_cache
def create_api_rate_limiter(
    max_requests: int,
    window_seconds: float,
) -> LLMAPIRateLimiter:
    """Return a shared configured API rate limiter."""

    return LLMAPIRateLimiter(
        max_requests=max_requests,
        window_seconds=window_seconds,
    )


def get_api_rate_limiter() -> LLMAPIRateLimiter:
    """Return the governed LLM API rate limiter."""

    settings = get_settings()

    return create_api_rate_limiter(
        settings.llm_api_rate_limit_requests,
        settings.llm_api_rate_limit_window_seconds,
    )


def get_governed_llm_api_service() -> GovernedLLMAPIService:
    """Return the configured governed LLM API service."""

    raise RuntimeError("Governed LLM API service dependency is not configured.")


def resolve_rate_limit_identity(
    *,
    request: Request,
    identity: AuthenticatedIdentity,
) -> str:
    """Resolve the trusted identity used for rate limiting."""

    if identity.authenticated:
        return identity.subject

    settings = get_settings()

    return resolve_client_identity(
        request,
        trusted_proxy_hosts=(settings.llm_api_trusted_proxy_hosts),
    )


GovernedLLMAPIServiceDependency = Annotated[
    GovernedLLMAPIService,
    Depends(get_governed_llm_api_service),
]

InFlightRequestGuardDependency = Annotated[
    InFlightRequestGuard,
    Depends(get_in_flight_request_guard),
]

LLMAPIRateLimiterDependency = Annotated[
    LLMAPIRateLimiter,
    Depends(get_api_rate_limiter),
]


@router.post(
    "/generate",
    response_model=LLMGenerateResponse,
)
async def generate(
    request_body: LLMGenerateRequest,
    request: Request,
    response: Response,
    service: GovernedLLMAPIServiceDependency,
    concurrency_guard: InFlightRequestGuardDependency,
    rate_limiter: LLMAPIRateLimiterDependency,
    identity: LLMGenerateIdentityDependency,
) -> LLMGenerateResponse:
    """Execute one governed LLM generation request."""

    correlation_id = (
        request_body.correlation_id
        or request.headers.get(CORRELATION_ID_HEADER)
        or create_correlation_id()
    )

    response.headers[CORRELATION_ID_HEADER] = correlation_id

    request_body = request_body.model_copy(
        update={
            "correlation_id": correlation_id,
        }
    )

    started_at = time.perf_counter()

    try:
        settings = get_settings()

        AuthenticationPolicy(
            authentication_required=(settings.llm_api_authentication_required)
        ).enforce(identity)

        validate_prompt_length(
            request_body.prompt,
            max_chars=settings.llm_api_max_prompt_chars,
        )

        caller_id = resolve_rate_limit_identity(
            request=request,
            identity=identity,
        )

        await rate_limiter.check(caller_id)

        async with concurrency_guard.slot():
            try:
                result = await asyncio.wait_for(
                    service.generate(request_body),
                    timeout=(settings.llm_api_request_timeout_seconds),
                )
            except TimeoutError as error:
                raise LLMRequestTimeoutError from error

        latency_ms = (time.perf_counter() - started_at) * 1000

        api_metrics.record_request(
            status_code=200,
            latency_ms=latency_ms,
            success=True,
            failure_code=None,
        )

        log_llm_api_request_event(
            create_llm_api_request_event(
                correlation_id=correlation_id,
                status_code=200,
                latency_ms=latency_ms,
                success=True,
                policy_identifier=result.policy_identifier,
                model=result.model,
                provider=result.provider,
                failure_code=None,
            )
        )

        return result

    except Exception as error:
        mapped = map_llm_exception(error)

        latency_ms = (time.perf_counter() - started_at) * 1000

        api_metrics.record_request(
            status_code=mapped.status_code,
            latency_ms=latency_ms,
            success=False,
            failure_code=mapped.code,
        )

        log_llm_api_request_event(
            create_llm_api_request_event(
                correlation_id=correlation_id,
                status_code=mapped.status_code,
                latency_ms=latency_ms,
                success=False,
                policy_identifier=None,
                model=None,
                provider=None,
                failure_code=mapped.code,
            )
        )

        http_error = mapped.to_http_exception()

        http_error.headers = {
            CORRELATION_ID_HEADER: correlation_id,
        }

        raise http_error from error
