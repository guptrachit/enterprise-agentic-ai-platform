import json
from dataclasses import dataclass

from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)


@dataclass(frozen=True)
class RoutingPolicyRecordSnapshot:
    """Immutable snapshot of one governed routing policy."""

    identifier: str
    policy_name: str
    version: str
    status: str
    description: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable policy record."""

        return {
            "identifier": self.identifier,
            "policy_name": self.policy_name,
            "version": self.version,
            "status": self.status,
            "description": self.description,
        }


@dataclass(frozen=True)
class RoutingPolicyActivationAuditSnapshot:
    """Immutable serialized view of one activation audit event."""

    timestamp: str
    policy_name: str
    activated_policy_identifier: str
    activated_version: str
    retired_policy_identifier: str | None
    retired_version: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable activation audit record."""

        return {
            "timestamp": self.timestamp,
            "policy_name": self.policy_name,
            "activated_policy_identifier": (self.activated_policy_identifier),
            "activated_version": self.activated_version,
            "retired_policy_identifier": (self.retired_policy_identifier),
            "retired_version": self.retired_version,
        }


@dataclass(frozen=True)
class RoutingPolicyRegistrySnapshot:
    """Immutable point-in-time snapshot of the routing policy registry."""

    policies: tuple[RoutingPolicyRecordSnapshot, ...]
    activation_history: tuple[RoutingPolicyActivationAuditSnapshot, ...]

    @property
    def policy_count(self) -> int:
        """Return the number of registered policies."""

        return len(self.policies)

    @property
    def activation_count(self) -> int:
        """Return the number of recorded activations."""

        return len(self.activation_history)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable registry snapshot."""

        return {
            "policy_count": self.policy_count,
            "activation_count": self.activation_count,
            "policies": [policy.to_dict() for policy in self.policies],
            "activation_history": [
                event.to_dict() for event in self.activation_history
            ],
        }

    def to_json(self) -> str:
        """Return the registry snapshot as stable JSON."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
        )


def create_registry_snapshot(
    registry: RoutingPolicyRegistry,
) -> RoutingPolicyRegistrySnapshot:
    """Create an immutable point-in-time registry snapshot."""

    policies = tuple(
        RoutingPolicyRecordSnapshot(
            identifier=policy.identifier,
            policy_name=policy.policy_name,
            version=policy.version,
            status=policy.status.value,
            description=(policy.versioned_policy.metadata.description),
        )
        for policy in registry.list_all()
    )

    activation_history = tuple(
        RoutingPolicyActivationAuditSnapshot(
            timestamp=event.timestamp,
            policy_name=event.policy_name,
            activated_policy_identifier=(event.activated_policy_identifier),
            activated_version=event.activated_version,
            retired_policy_identifier=(event.retired_policy_identifier),
            retired_version=event.retired_version,
        )
        for event in registry.activation_history()
    )

    return RoutingPolicyRegistrySnapshot(
        policies=policies,
        activation_history=activation_history,
    )
