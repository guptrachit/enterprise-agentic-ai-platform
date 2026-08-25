from agent_platform.config import Settings
from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)


def create_configured_routing_policy_registry(
    settings: Settings,
) -> RoutingPolicyRegistry:
    """Create a governed routing-policy registry from application settings."""

    policy = ModelPolicy(assignments=settings.llm_model_policy.assignments)

    versioned_policy = VersionedRoutingPolicy(
        policy=policy,
        metadata=RoutingPolicyMetadata(
            name=settings.llm_routing_policy_name,
            version=RoutingPolicyVersion(
                major=1,
                minor=0,
                patch=0,
            ),
            description=("Routing policy bootstrapped from application settings."),
        ),
    )

    governed_policy = GovernedRoutingPolicy(
        versioned_policy=versioned_policy,
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry = RoutingPolicyRegistry()
    registry.register(governed_policy)

    return registry
