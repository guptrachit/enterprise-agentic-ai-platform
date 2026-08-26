from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_platform.config import get_settings
from agent_platform.llm.api_health import (
    create_llm_api_health_payload,
)
from agent_platform.llm.api_router import (
    api_metrics,
    get_api_rate_limiter,
    get_governed_llm_api_service,
    get_in_flight_request_guard,
)
from agent_platform.llm.api_router import (
    router as llm_router,
)
from agent_platform.llm.api_service import GovernedLLMAPIService
from agent_platform.llm.factory import create_llm_client_for_model
from agent_platform.llm.fully_configured_governed_runtime import (
    create_fully_configured_governed_runtime,
)
from agent_platform.observability.operational_service import (
    OperationalObservabilityService,
)
from agent_platform.security.content_type_middleware import (
    JSONContentTypeMiddleware,
)
from agent_platform.security.exception_handlers import (
    register_security_exception_handlers,
    security_metrics,
)
from agent_platform.security.request_size_middleware import (
    RequestBodySizeMiddleware,
)
from agent_platform.security.security_headers import (
    SecurityHeadersMiddleware,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """Create application-scoped governed LLM runtime."""

    settings = get_settings()

    runtime = create_fully_configured_governed_runtime(
        settings=settings,
        client_factory=lambda model: create_llm_client_for_model(
            settings,
            model,
        ),
    )

    app.state.governed_llm_runtime = runtime

    app.state.governed_llm_api_service = GovernedLLMAPIService(
        runtime=runtime.refresh_service
    )

    yield


settings = get_settings()

app = FastAPI(
    title="Enterprise Agentic AI Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.llm_api_cors_allowed_origins),
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

app.add_middleware(JSONContentTypeMiddleware)

app.add_middleware(
    RequestBodySizeMiddleware,
    max_body_bytes=(settings.llm_api_max_request_body_bytes),
)

app.add_middleware(SecurityHeadersMiddleware)

register_security_exception_handlers(app)


def get_application_llm_api_service() -> GovernedLLMAPIService:
    """Return the application-scoped governed LLM API service."""

    return app.state.governed_llm_api_service


app.dependency_overrides[get_governed_llm_api_service] = get_application_llm_api_service

app.include_router(llm_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "application": "Enterprise Agentic AI Platform",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@app.get("/health/llm")
def llm_health_check() -> dict[str, object]:
    """Return unified governed LLM operational health."""

    runtime = app.state.governed_llm_runtime

    observability_service = OperationalObservabilityService(
        runtime=runtime.refresh_service,
        api_metrics=api_metrics,
        concurrency_guard=get_in_flight_request_guard(),
        rate_limiter=get_api_rate_limiter(),
        security_metrics=security_metrics,
    )

    return create_llm_api_health_payload(
        observability_service=observability_service,
    )
