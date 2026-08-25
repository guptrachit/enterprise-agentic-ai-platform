from dataclasses import dataclass

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
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


@dataclass(frozen=True)
class GovernedRuntimeBootstrap:
    """Fully wired governed LLM runtime components."""

    registry: RoutingPolicyRegistry
    resolver: ActiveRoutingPolicyResolver
    router_factory: RuntimeModelRouterFactory
    execution_service_factory: RuntimeLLMExecutionServiceFactory
    refresh_service: RefreshAwareLLMExecutionService
    refresh_metrics: RuntimePolicyRefreshMetrics | None
