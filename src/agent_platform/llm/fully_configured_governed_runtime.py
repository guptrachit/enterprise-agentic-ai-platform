from collections.abc import Callable

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient
from agent_platform.llm.configured_routing_policy_registry import (
    create_configured_routing_policy_registry,
)
from agent_platform.llm.governed_runtime_bootstrap import (
    GovernedRuntimeBootstrap,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_registry_loader import (
    load_model_registry,
)
from agent_platform.llm.routing_metrics import RoutingMetrics
from agent_platform.llm.routing_metrics_exporter import (
    RoutingMetricsExporter,
)
from agent_platform.llm.settings_driven_governed_runtime import (
    create_settings_driven_governed_runtime,
)


def create_fully_configured_governed_runtime(
    *,
    settings: Settings,
    client_factory: Callable[[ModelDefinition], LLMClient],
    routing_metrics: RoutingMetrics | None = None,
    routing_metrics_exporter: RoutingMetricsExporter | None = None,
) -> GovernedRuntimeBootstrap:
    """Create a fully configured governed runtime from application settings."""

    model_registry = load_model_registry(settings)

    registry = create_configured_routing_policy_registry(settings)

    return create_settings_driven_governed_runtime(
        settings=settings,
        registry=registry,
        models=model_registry.as_dict(),
        client_factory=client_factory,
        routing_metrics=routing_metrics,
        routing_metrics_exporter=routing_metrics_exporter,
    )
