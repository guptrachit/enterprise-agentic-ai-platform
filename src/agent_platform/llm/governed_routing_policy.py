from dataclasses import dataclass

from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)


@dataclass(frozen=True)
class GovernedRoutingPolicy:
    """Versioned routing policy with lifecycle governance state."""

    versioned_policy: VersionedRoutingPolicy
    status: RoutingPolicyLifecycleStatus

    @property
    def identifier(self) -> str:
        """Return the governed policy identifier."""

        return self.versioned_policy.identifier

    @property
    def policy_name(self) -> str:
        """Return the routing policy name."""

        return self.versioned_policy.metadata.name

    @property
    def version(self) -> str:
        """Return the routing policy version."""

        return str(self.versioned_policy.metadata.version)

    @property
    def is_active(self) -> bool:
        """Return whether this policy is currently active."""

        return self.status is RoutingPolicyLifecycleStatus.ACTIVE

    @property
    def is_retired(self) -> bool:
        """Return whether this policy has been retired."""

        return self.status is RoutingPolicyLifecycleStatus.RETIRED
