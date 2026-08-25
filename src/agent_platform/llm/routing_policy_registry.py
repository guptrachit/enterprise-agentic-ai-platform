from dataclasses import dataclass

from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.routing_policy_activation_audit import (
    RoutingPolicyActivationAuditEvent,
    create_activation_audit_event,
)
from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.routing_policy_lifecycle_transition import (
    RoutingPolicyLifecycleTransitionResult,
    evaluate_lifecycle_transition,
)


@dataclass(frozen=True)
class RoutingPolicyActivationResult:
    """Result of activating an approved routing policy version."""

    activated_policy: GovernedRoutingPolicy
    retired_policy: GovernedRoutingPolicy | None


class RoutingPolicyRegistry:
    """In-memory registry of governed routing policy versions."""

    def __init__(self) -> None:
        self._policies: dict[str, GovernedRoutingPolicy] = {}
        self._activation_history: list[RoutingPolicyActivationAuditEvent] = []

    def register(
        self,
        policy: GovernedRoutingPolicy,
    ) -> None:
        """Register one governed routing policy."""

        identifier = policy.identifier

        if identifier in self._policies:
            raise ValueError(f"Routing policy '{identifier}' is already registered.")

        if (
            policy.status == RoutingPolicyLifecycleStatus.ACTIVE
            and self._has_active_policy(policy.policy_name)
        ):
            raise ValueError(
                f"Routing policy '{policy.policy_name}' already has an active version."
            )

        self._policies[identifier] = policy

    def get(
        self,
        identifier: str,
    ) -> GovernedRoutingPolicy:
        """Return a governed routing policy by identifier."""

        try:
            return self._policies[identifier]
        except KeyError as error:
            raise KeyError(
                f"Routing policy '{identifier}' is not registered."
            ) from error

    def contains(
        self,
        identifier: str,
    ) -> bool:
        """Return whether a governed routing policy is registered."""

        return identifier in self._policies

    def list_all(
        self,
    ) -> tuple[GovernedRoutingPolicy, ...]:
        """Return all registered policies in registration order."""

        return tuple(self._policies.values())

    def find_by_name(
        self,
        policy_name: str,
    ) -> tuple[GovernedRoutingPolicy, ...]:
        """Return all registered versions of a policy name."""

        return tuple(
            policy
            for policy in self._policies.values()
            if policy.policy_name == policy_name
        )

    def find_by_status(
        self,
        status: RoutingPolicyLifecycleStatus,
    ) -> tuple[GovernedRoutingPolicy, ...]:
        """Return all policies with the requested lifecycle status."""

        return tuple(
            policy for policy in self._policies.values() if policy.status == status
        )

    def find_by_name_and_status(
        self,
        *,
        policy_name: str,
        status: RoutingPolicyLifecycleStatus,
    ) -> tuple[GovernedRoutingPolicy, ...]:
        """Return matching policies for a name and lifecycle status."""

        return tuple(
            policy
            for policy in self._policies.values()
            if (policy.policy_name == policy_name and policy.status == status)
        )

    def get_active(
        self,
        policy_name: str,
    ) -> GovernedRoutingPolicy | None:
        """Return the active version of a policy name, if one exists."""

        active = self._active_policies(policy_name)

        if not active:
            return None

        if len(active) > 1:
            raise RuntimeError(
                f"Multiple active routing policies found for '{policy_name}'."
            )

        return active[0]

    def transition(
        self,
        *,
        identifier: str,
        target_status: RoutingPolicyLifecycleStatus,
    ) -> RoutingPolicyLifecycleTransitionResult:
        """Apply a valid lifecycle transition to a registered policy."""

        current = self.get(identifier)

        result = evaluate_lifecycle_transition(
            current_status=current.status,
            target_status=target_status,
        )

        if not result.allowed:
            return result

        if target_status == RoutingPolicyLifecycleStatus.ACTIVE:
            active_policies = self._active_policies(current.policy_name)

            conflicting_active = tuple(
                policy for policy in active_policies if policy.identifier != identifier
            )

            if conflicting_active:
                raise ValueError(
                    f"Routing policy '{current.policy_name}' "
                    "already has an active version."
                )

        self._policies[identifier] = GovernedRoutingPolicy(
            versioned_policy=current.versioned_policy,
            status=target_status,
        )

        return result

    def activate(
        self,
        identifier: str,
    ) -> RoutingPolicyActivationResult:
        """Atomically activate an approved routing policy version."""

        candidate = self.get(identifier)

        activation_transition = evaluate_lifecycle_transition(
            current_status=candidate.status,
            target_status=RoutingPolicyLifecycleStatus.ACTIVE,
        )

        if not activation_transition.allowed:
            raise ValueError(
                f"Routing policy '{identifier}' cannot be activated "
                f"from lifecycle status '{candidate.status.value}'."
            )

        current_active = self.get_active(candidate.policy_name)

        if current_active is not None:
            retirement_transition = evaluate_lifecycle_transition(
                current_status=current_active.status,
                target_status=RoutingPolicyLifecycleStatus.RETIRED,
            )

            if not retirement_transition.allowed:
                raise ValueError(
                    f"Active routing policy "
                    f"'{current_active.identifier}' cannot be retired."
                )

        retired_policy = None

        if current_active is not None:
            retired_policy = GovernedRoutingPolicy(
                versioned_policy=current_active.versioned_policy,
                status=RoutingPolicyLifecycleStatus.RETIRED,
            )

        activated_policy = GovernedRoutingPolicy(
            versioned_policy=candidate.versioned_policy,
            status=RoutingPolicyLifecycleStatus.ACTIVE,
        )

        if retired_policy is not None:
            self._policies[retired_policy.identifier] = retired_policy

        self._policies[activated_policy.identifier] = activated_policy

        result = RoutingPolicyActivationResult(
            activated_policy=activated_policy,
            retired_policy=retired_policy,
        )

        self._activation_history.append(create_activation_audit_event(result))

        return result

    def activation_history(
        self,
    ) -> tuple[RoutingPolicyActivationAuditEvent, ...]:
        """Return all activation audit events in activation order."""

        return tuple(self._activation_history)

    def activation_history_for(
        self,
        policy_name: str,
    ) -> tuple[RoutingPolicyActivationAuditEvent, ...]:
        """Return activation audit history for one policy name."""

        return tuple(
            event
            for event in self._activation_history
            if event.policy_name == policy_name
        )

    def _active_policies(
        self,
        policy_name: str,
    ) -> tuple[GovernedRoutingPolicy, ...]:
        """Return active versions for one policy name."""

        return tuple(
            policy
            for policy in self._policies.values()
            if (
                policy.policy_name == policy_name
                and policy.status == RoutingPolicyLifecycleStatus.ACTIVE
            )
        )

    def _has_active_policy(
        self,
        policy_name: str,
    ) -> bool:
        """Return whether a policy name already has an active version."""

        return bool(self._active_policies(policy_name))

    def __len__(self) -> int:
        """Return the number of registered policies."""

        return len(self._policies)
