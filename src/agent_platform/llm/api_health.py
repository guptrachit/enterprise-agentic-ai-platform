from agent_platform.llm.api_metrics import (
    LLMAPIMetrics,
)
from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)


def create_llm_api_health_payload(
    *,
    runtime: RefreshAwareLLMExecutionService,
    api_metrics: LLMAPIMetrics,
) -> dict[str, object]:
    """Create a safe operational health payload for the governed LLM API."""

    return {
        "status": "healthy",
        "runtime": runtime.health_snapshot().to_dict(),
        "api_metrics": api_metrics.snapshot().to_dict(),
    }
