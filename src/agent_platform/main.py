from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_platform.config import get_settings
from agent_platform.llm.api_health import (
    create_llm_api_health_payload,
)
from agent_platform.llm.api_router import (
    api_metrics,
    get_governed_llm_api_service,
)
from agent_platform.llm.api_router import (
    router as llm_router,
)
from agent_platform.llm.api_service import GovernedLLMAPIService
from agent_platform.llm.factory import create_llm_client_for_model
from agent_platform.llm.fully_configured_governed_runtime import (
    create_fully_configured_governed_runtime,
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


app = FastAPI(
    title="Enterprise Agentic AI Platform",
    version="0.1.0",
    lifespan=lifespan,
)


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
    return {"status": "healthy"}


@app.get("/health/llm")
def llm_health_check() -> dict[str, object]:
    """Return safe governed LLM runtime operational health."""

    runtime = app.state.governed_llm_runtime

    return create_llm_api_health_payload(
        runtime=runtime.refresh_service,
        api_metrics=api_metrics,
    )
