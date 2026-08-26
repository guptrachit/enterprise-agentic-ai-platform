import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from agent_platform.correlation import (
    CORRELATION_ID_HEADER,
    create_correlation_id,
)
from agent_platform.llm.api_errors import (
    map_llm_exception,
)
from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)
from agent_platform.llm.api_models import (
    LLMGenerateRequest,
    LLMGenerateResponse,
)
from agent_platform.llm.api_service import (
    GovernedLLMAPIService,
)
from agent_platform.llm.api_telemetry import (
    create_llm_api_request_event,
    log_llm_api_request_event,
)

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)

api_metrics = LLMAPIMetrics()


def get_governed_llm_api_service() -> GovernedLLMAPIService:
    """Return the configured governed LLM API service."""

    raise RuntimeError("Governed LLM API service dependency is not configured.")


GovernedLLMAPIServiceDependency = Annotated[
    GovernedLLMAPIService,
    Depends(get_governed_llm_api_service),
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
        result = await service.generate(request_body)

        latency_ms = (time.perf_counter() - started_at) * 1000

        api_metrics.record_request(
            status_code=200,
            latency_ms=latency_ms,
            success=True,
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
            )
        )

        raise mapped.to_http_exception() from error
