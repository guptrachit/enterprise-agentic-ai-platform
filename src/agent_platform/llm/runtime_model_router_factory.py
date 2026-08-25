from collections.abc import Mapping
from dataclasses import dataclass

from agent_platform.llm.active_routing_policy_resolver import (
    ActiveRoutingPolicyResolver,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_router import ModelRouter


@dataclass(frozen=True)
class RuntimeModelRouterResolution:
    """Resolved runtime model router with governed policy identity."""

    router: ModelRouter
    policy_identifier: str


class RuntimeModelRouterFactory:
    """Build runtime model routers from the currently active policy."""

    def __init__(
        self,
        *,
        resolver: ActiveRoutingPolicyResolver,
        models: Mapping[str, ModelDefinition],
    ) -> None:
        self.resolver = resolver
        self.models = dict(models)

    def resolve(
        self,
        policy_name: str,
    ) -> RuntimeModelRouterResolution:
        """Resolve the active policy and build its runtime router."""

        versioned_policy = self.resolver.resolve(policy_name)

        router = ModelRouter(
            models=self.models,
            policy=versioned_policy.policy,
        )

        return RuntimeModelRouterResolution(
            router=router,
            policy_identifier=versioned_policy.identifier,
        )

    def create(
        self,
        policy_name: str,
    ) -> ModelRouter:
        """Create a ModelRouter using the active governed policy."""

        return self.resolve(policy_name).router
