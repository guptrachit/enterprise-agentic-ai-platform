from unittest.mock import Mock

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
from agent_platform.llm.governed_runtime_bootstrap import (
    GovernedRuntimeBootstrap,
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


def test_governed_runtime_bootstrap_preserves_components() -> None:
    registry = RoutingPolicyRegistry()

    resolver = ActiveRoutingPolicyResolver(registry)

    router_factory = Mock(
        spec=RuntimeModelRouterFactory,
    )

    execution_service_factory = Mock(
        spec=RuntimeLLMExecutionServiceFactory,
    )

    refresh_service = Mock(
        spec=RefreshAwareLLMExecutionService,
    )

    refresh_metrics = RuntimePolicyRefreshMetrics()

    runtime = GovernedRuntimeBootstrap(
        registry=registry,
        resolver=resolver,
        router_factory=router_factory,
        execution_service_factory=execution_service_factory,
        refresh_service=refresh_service,
        refresh_metrics=refresh_metrics,
    )

    assert runtime.registry is registry
    assert runtime.resolver is resolver
    assert runtime.router_factory is router_factory

    assert runtime.execution_service_factory is execution_service_factory

    assert runtime.refresh_service is refresh_service
    assert runtime.refresh_metrics is refresh_metrics


def test_governed_runtime_bootstrap_allows_metrics_disabled() -> None:
    registry = RoutingPolicyRegistry()

    runtime = GovernedRuntimeBootstrap(
        registry=registry,
        resolver=ActiveRoutingPolicyResolver(registry),
        router_factory=Mock(
            spec=RuntimeModelRouterFactory,
        ),
        execution_service_factory=Mock(
            spec=RuntimeLLMExecutionServiceFactory,
        ),
        refresh_service=Mock(
            spec=RefreshAwareLLMExecutionService,
        ),
        refresh_metrics=None,
    )

    assert runtime.refresh_metrics is None
