from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)


class ActiveRoutingPolicyResolver:
    """Resolve the currently active governed routing policy."""

    def __init__(
        self,
        registry: RoutingPolicyRegistry,
    ) -> None:
        self.registry = registry

    def resolve(
        self,
        policy_name: str,
    ) -> VersionedRoutingPolicy:
        """Resolve the active versioned routing policy by name."""

        active = self.registry.get_active(policy_name)

        if active is None:
            raise LookupError(f"No active routing policy found for '{policy_name}'.")

        return active.versioned_policy
