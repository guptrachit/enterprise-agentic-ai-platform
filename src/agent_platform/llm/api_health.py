from agent_platform.observability.operational_service import (
    OperationalObservabilityService,
)


def create_llm_api_health_payload(
    *,
    observability_service: OperationalObservabilityService,
) -> dict[str, object]:
    """Create safe operational health from the unified snapshot."""

    snapshot = observability_service.snapshot()

    return snapshot.to_dict()
