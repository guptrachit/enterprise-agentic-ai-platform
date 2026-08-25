from collections.abc import Callable, Mapping

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
from agent_platform.llm.base import LLMClient
from agent_platform.llm.governed_runtime_bootstrap import (
    GovernedRuntimeBootstrap,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)
from agent_platform.llm.routing_metrics import RoutingMetrics
from agent_platform.llm.routing_metrics_exporter import (
    RoutingMetricsExporter,
)
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)
from agent_platform.llm.runtime_governance_config import (
    RuntimeGovernanceConfig,
)
from agent_platform.llm.runtime_llm_execution_service_factory import (
    RuntimeLLMExecutionServiceFactory,
)
from agent_platform.llm.runtime_model_router_factory import (
    RuntimeModelRouterFactory,
)
from agent_platform.llm.runtime_policy_refresh_metrics import (
    RuntimePolicyRefreshMetrics,
)


def create_governed_runtime(
    *,
    registry: RoutingPolicyRegistry,
    models: Mapping[str, ModelDefinition],
    client_factory: Callable[[ModelDefinition], LLMClient],
    config: RuntimeGovernanceConfig,
    routing_metrics: RoutingMetrics | None = None,
    routing_metrics_exporter: RoutingMetricsExporter | None = None,
) -> GovernedRuntimeBootstrap:
    """Create a fully wired governed LLM runtime."""

    resolver = ActiveRoutingPolicyResolver(registry)

    router_factory = RuntimeModelRouterFactory(
        resolver=resolver,
        models=models,
    )

    execution_service_factory = RuntimeLLMExecutionServiceFactory(
        router_factory=router_factory,
        client_factory=client_factory,
        metrics=routing_metrics,
        metrics_exporter=routing_metrics_exporter,
    )

    refresh_metrics = (
        RuntimePolicyRefreshMetrics() if config.enable_refresh_metrics else None
    )

    refresh_service = RefreshAwareLLMExecutionService(
        policy_name=config.policy_name,
        factory=execution_service_factory,
        refresh_config=config.refresh,
        refresh_metrics=refresh_metrics,
    )

    return GovernedRuntimeBootstrap(
        registry=registry,
        resolver=resolver,
        router_factory=router_factory,
        execution_service_factory=execution_service_factory,
        refresh_service=refresh_service,
        refresh_metrics=refresh_metrics,
    )
