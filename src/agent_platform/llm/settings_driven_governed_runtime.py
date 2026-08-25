from collections.abc import Callable, Mapping

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient
from agent_platform.llm.governed_runtime_bootstrap import (
    GovernedRuntimeBootstrap,
)
from agent_platform.llm.governed_runtime_factory import (
    create_governed_runtime,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.routing_metrics import RoutingMetrics
from agent_platform.llm.routing_metrics_exporter import (
    RoutingMetricsExporter,
)
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)
from agent_platform.llm.runtime_governance_config_factory import (
    create_runtime_governance_config,
)


def create_settings_driven_governed_runtime(
    *,
    settings: Settings,
    registry: RoutingPolicyRegistry,
    models: Mapping[str, ModelDefinition],
    client_factory: Callable[[ModelDefinition], LLMClient],
    routing_metrics: RoutingMetrics | None = None,
    routing_metrics_exporter: RoutingMetricsExporter | None = None,
) -> GovernedRuntimeBootstrap:
    """Create the governed runtime directly from application settings."""

    config = create_runtime_governance_config(settings)

    return create_governed_runtime(
        registry=registry,
        models=models,
        client_factory=client_factory,
        config=config,
        routing_metrics=routing_metrics,
        routing_metrics_exporter=routing_metrics_exporter,
    )
