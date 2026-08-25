from dataclasses import dataclass

from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyRegistry,
)


@dataclass(frozen=True)
class RoutingPolicyRegistryValidationResult:
    """Validation result for routing policy registry invariants."""

    valid: bool
    errors: tuple[str, ...]

    @property
    def error_count(self) -> int:
        """Return the number of validation errors."""

        return len(self.errors)


def validate_routing_policy_registry(
    registry: RoutingPolicyRegistry,
) -> RoutingPolicyRegistryValidationResult:
    """Validate routing policy registry invariants."""

    errors: list[str] = []

    policies = registry.list_all()

    identifiers = tuple(policy.identifier for policy in policies)

    if len(set(identifiers)) != len(identifiers):
        errors.append("Registry contains duplicate routing policy identifiers.")

    policy_names = {policy.policy_name for policy in policies}

    for policy_name in policy_names:
        active = registry.find_by_name_and_status(
            policy_name=policy_name,
            status=RoutingPolicyLifecycleStatus.ACTIVE,
        )

        if len(active) > 1:
            errors.append(
                f"Routing policy '{policy_name}' has multiple active versions."
            )

    registered_identifiers = set(identifiers)

    for event in registry.activation_history():
        if event.activated_policy_identifier not in registered_identifiers:
            errors.append(
                f"Activation history references unregistered "
                f"activated policy "
                f"'{event.activated_policy_identifier}'."
            )

        if (
            event.retired_policy_identifier is not None
            and event.retired_policy_identifier not in registered_identifiers
        ):
            errors.append(
                f"Activation history references unregistered "
                f"retired policy "
                f"'{event.retired_policy_identifier}'."
            )

        if event.activated_policy_identifier in registered_identifiers:
            activated = registry.get(event.activated_policy_identifier)

            if activated.status not in (
                RoutingPolicyLifecycleStatus.ACTIVE,
                RoutingPolicyLifecycleStatus.RETIRED,
            ):
                errors.append(
                    f"Activated policy "
                    f"'{event.activated_policy_identifier}' "
                    f"has inconsistent lifecycle status "
                    f"'{activated.status.value}'."
                )

    return RoutingPolicyRegistryValidationResult(
        valid=not errors,
        errors=tuple(errors),
    )
