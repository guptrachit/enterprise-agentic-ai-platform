from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)


def runtime_policy_health_payload(
    service: RefreshAwareLLMExecutionService,
) -> dict[str, object]:
    """Return a JSON-safe runtime routing-policy health payload."""

    return service.health_snapshot().to_dict()
