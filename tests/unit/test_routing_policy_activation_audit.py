from agent_platform.llm.governed_routing_policy import (
    GovernedRoutingPolicy,
)
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_policy_activation_audit import (
    RoutingPolicyActivationAuditEvent,
    create_activation_audit_event,
)
from agent_platform.llm.routing_policy_lifecycle import (
    RoutingPolicyLifecycleStatus,
)
from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)
from agent_platform.llm.routing_policy_registry import (
    RoutingPolicyActivationResult,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def create_policy(
    *,
    version: str,
    status: RoutingPolicyLifecycleStatus,
) -> GovernedRoutingPolicy:
    model_policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "primary",
        }
    )

    versioned_policy = VersionedRoutingPolicy(
        policy=model_policy,
        metadata=RoutingPolicyMetadata(
            name="production-policy",
            version=RoutingPolicyVersion.parse(version),
        ),
    )

    return GovernedRoutingPolicy(
        versioned_policy=versioned_policy,
        status=status,
    )


def test_activation_audit_event_without_retired_policy() -> None:
    activated = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    result = RoutingPolicyActivationResult(
        activated_policy=activated,
        retired_policy=None,
    )

    event = create_activation_audit_event(result)

    assert isinstance(
        event,
        RoutingPolicyActivationAuditEvent,
    )

    assert event.policy_name == "production-policy"

    assert event.activated_policy_identifier == ("production-policy@1.0.0")

    assert event.activated_version == "1.0.0"

    assert event.retired_policy_identifier is None
    assert event.retired_version is None
    assert event.timestamp


def test_activation_audit_event_with_retired_policy() -> None:
    retired = create_policy(
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.RETIRED,
    )

    activated = create_policy(
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    result = RoutingPolicyActivationResult(
        activated_policy=activated,
        retired_policy=retired,
    )

    event = create_activation_audit_event(result)

    assert event.policy_name == "production-policy"

    assert event.activated_policy_identifier == ("production-policy@1.1.0")

    assert event.activated_version == "1.1.0"

    assert event.retired_policy_identifier == ("production-policy@1.0.0")

    assert event.retired_version == "1.0.0"


def test_activation_audit_event_is_immutable() -> None:
    activated = create_policy(
        version="2.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    event = create_activation_audit_event(
        RoutingPolicyActivationResult(
            activated_policy=activated,
            retired_policy=None,
        )
    )

    try:
        event.activated_version = "9.9.9"
    except Exception:
        pass

    assert event.activated_version == "2.0.0"
