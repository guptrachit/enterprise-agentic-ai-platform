from collections.abc import Callable

from agent_platform.llm.base import LLMClient
from agent_platform.llm.execution_service import LLMExecutionService
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.routing_metrics import RoutingMetrics
from agent_platform.llm.routing_metrics_exporter import (
    RoutingMetricsExporter,
)
from agent_platform.llm.runtime_model_router_factory import (
    RuntimeModelRouterFactory,
)


class RuntimeLLMExecutionServiceFactory:
    """Build governed runtime LLM execution services."""

    def __init__(
        self,
        *,
        router_factory: RuntimeModelRouterFactory,
        client_factory: Callable[
            [ModelDefinition],
            LLMClient,
        ],
        metrics: RoutingMetrics | None = None,
        metrics_exporter: RoutingMetricsExporter | None = None,
    ) -> None:
        self.router_factory = router_factory
        self.client_factory = client_factory
        self.metrics = metrics
        self.metrics_exporter = metrics_exporter

    def create(
        self,
        policy_name: str,
    ) -> LLMExecutionService:
        """Create an execution service from the active governed policy."""

        resolution = self.router_factory.resolve(policy_name)

        return LLMExecutionService(
            router=resolution.router,
            client_factory=self.client_factory,
            metrics=self.metrics,
            metrics_exporter=self.metrics_exporter,
            policy_identifier=resolution.policy_identifier,
        )
