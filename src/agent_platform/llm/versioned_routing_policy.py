from dataclasses import dataclass

from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)


@dataclass(frozen=True)
class VersionedRoutingPolicy:
    """Model routing policy paired with governance metadata."""

    policy: ModelPolicy
    metadata: RoutingPolicyMetadata

    @property
    def identifier(self) -> str:
        """Return the governed policy identifier."""

        return self.metadata.identifier
