import pytest

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
from agent_platform.llm.workload import LLMWorkload


def create_governed_policy(
    *,
    name: str = "production-routing-policy",
    version: str = "1.0.0",
    status: RoutingPolicyLifecycleStatus = (RoutingPolicyLifecycleStatus.DRAFT),
) -> GovernedRoutingPolicy:
    model_policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "primary",
        }
    )

    versioned_policy = VersionedRoutingPolicy(
        policy=model_policy,
        metadata=RoutingPolicyMetadata(
            name=name,
            version=RoutingPolicyVersion.parse(version),
        ),
    )

    return GovernedRoutingPolicy(
        versioned_policy=versioned_policy,
        status=status,
    )


def test_registry_starts_empty() -> None:
    registry = RoutingPolicyRegistry()

    assert len(registry) == 0
    assert registry.list_all() == ()


def test_registry_registers_policy() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy()

    registry.register(policy)

    assert len(registry) == 1

    assert registry.contains("production-routing-policy@1.0.0")


def test_registry_returns_registered_policy() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy()

    registry.register(policy)

    result = registry.get("production-routing-policy@1.0.0")

    assert result is policy


def test_registry_rejects_duplicate_identifier() -> None:
    registry = RoutingPolicyRegistry()

    first = create_governed_policy()

    duplicate = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    registry.register(first)

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(duplicate)


def test_registry_raises_for_missing_policy() -> None:
    registry = RoutingPolicyRegistry()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        registry.get("missing-policy@1.0.0")


def test_registry_lists_policies_in_registration_order() -> None:
    registry = RoutingPolicyRegistry()

    first = create_governed_policy(
        version="1.0.0",
    )

    second = create_governed_policy(
        version="1.1.0",
    )

    third = create_governed_policy(
        version="2.0.0",
    )

    registry.register(first)
    registry.register(second)
    registry.register(third)

    assert registry.list_all() == (
        first,
        second,
        third,
    )


def test_registry_supports_multiple_policy_names() -> None:
    registry = RoutingPolicyRegistry()

    production = create_governed_policy(
        name="production-policy",
        version="1.0.0",
    )

    experimental = create_governed_policy(
        name="experimental-policy",
        version="1.0.0",
    )

    registry.register(production)
    registry.register(experimental)

    assert len(registry) == 2

    assert registry.contains("production-policy@1.0.0")

    assert registry.contains("experimental-policy@1.0.0")


def test_find_by_name_returns_all_versions() -> None:
    registry = RoutingPolicyRegistry()

    first = create_governed_policy(
        name="production-policy",
        version="1.0.0",
    )

    second = create_governed_policy(
        name="production-policy",
        version="1.1.0",
    )

    other = create_governed_policy(
        name="experimental-policy",
        version="1.0.0",
    )

    registry.register(first)
    registry.register(second)
    registry.register(other)

    assert registry.find_by_name("production-policy") == (
        first,
        second,
    )


def test_find_by_name_returns_empty_for_missing_name() -> None:
    registry = RoutingPolicyRegistry()

    assert registry.find_by_name("missing-policy") == ()


def test_find_by_status_returns_matching_policies() -> None:
    registry = RoutingPolicyRegistry()

    active = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    other_active = create_governed_policy(
        name="experimental-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry.register(active)
    registry.register(candidate)
    registry.register(other_active)

    assert registry.find_by_status(RoutingPolicyLifecycleStatus.ACTIVE) == (
        active,
        other_active,
    )


def test_find_by_name_and_status_filters_both() -> None:
    registry = RoutingPolicyRegistry()

    active = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    registry.register(active)
    registry.register(candidate)

    assert registry.find_by_name_and_status(
        policy_name="production-policy",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    ) == (active,)


def test_get_active_returns_active_policy() -> None:
    registry = RoutingPolicyRegistry()

    active = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    registry.register(active)
    registry.register(candidate)

    assert registry.get_active("production-policy") is active


def test_get_active_returns_none_when_no_active_policy_exists() -> None:
    registry = RoutingPolicyRegistry()

    registry.register(
        create_governed_policy(
            name="production-policy",
            version="1.1.0",
            status=RoutingPolicyLifecycleStatus.CANDIDATE,
        )
    )

    assert registry.get_active("production-policy") is None


def test_registry_prevents_multiple_active_versions() -> None:
    registry = RoutingPolicyRegistry()

    first = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    second = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry.register(first)

    with pytest.raises(
        ValueError,
        match="already has an active version",
    ):
        registry.register(second)

    assert registry.get_active("production-policy") is first

    assert len(registry) == 1


def test_registry_transitions_policy_to_next_valid_status() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.DRAFT,
    )

    registry.register(policy)

    result = registry.transition(
        identifier=policy.identifier,
        target_status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    assert result.allowed is True

    updated = registry.get(policy.identifier)

    assert updated.status is RoutingPolicyLifecycleStatus.CANDIDATE

    assert updated.versioned_policy is policy.versioned_policy


def test_registry_transition_preserves_identifier() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy(
        version="1.2.0",
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    registry.register(policy)

    registry.transition(
        identifier=policy.identifier,
        target_status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    updated = registry.get(policy.identifier)

    assert updated.identifier == ("production-routing-policy@1.2.0")


def test_registry_rejects_invalid_transition_without_mutation() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.DRAFT,
    )

    registry.register(policy)

    result = registry.transition(
        identifier=policy.identifier,
        target_status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    assert result.allowed is False

    unchanged = registry.get(policy.identifier)

    assert unchanged.status is RoutingPolicyLifecycleStatus.DRAFT

    assert unchanged is policy


def test_registry_supports_complete_lifecycle() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.DRAFT,
    )

    registry.register(policy)

    for target_status in (
        RoutingPolicyLifecycleStatus.CANDIDATE,
        RoutingPolicyLifecycleStatus.APPROVED,
        RoutingPolicyLifecycleStatus.ACTIVE,
        RoutingPolicyLifecycleStatus.RETIRED,
    ):
        result = registry.transition(
            identifier=policy.identifier,
            target_status=target_status,
        )

        assert result.allowed is True

        assert registry.get(policy.identifier).status is target_status


def test_registry_rejects_transition_from_retired_policy() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.RETIRED,
    )

    registry.register(policy)

    result = registry.transition(
        identifier=policy.identifier,
        target_status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    assert result.allowed is False

    assert (
        registry.get(policy.identifier).status is RoutingPolicyLifecycleStatus.RETIRED
    )


def test_registry_transition_raises_for_missing_policy() -> None:
    registry = RoutingPolicyRegistry()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        registry.transition(
            identifier="missing-policy@1.0.0",
            target_status=(RoutingPolicyLifecycleStatus.CANDIDATE),
        )


def test_registry_queries_reflect_transitioned_status() -> None:
    registry = RoutingPolicyRegistry()

    policy = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(policy)

    assert registry.get_active(policy.policy_name) is None

    registry.transition(
        identifier=policy.identifier,
        target_status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    active = registry.get_active(policy.policy_name)

    assert active is not None
    assert active.identifier == policy.identifier


def test_registry_rejects_second_active_policy_on_registration() -> None:
    registry = RoutingPolicyRegistry()

    first = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    second = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry.register(first)

    with pytest.raises(
        ValueError,
        match="already has an active version",
    ):
        registry.register(second)

    assert len(registry) == 1


def test_registry_allows_active_policies_for_different_names() -> None:
    registry = RoutingPolicyRegistry()

    production = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    experimental = create_governed_policy(
        name="experimental-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    registry.register(production)
    registry.register(experimental)

    assert registry.get_active("production-policy") is production

    assert registry.get_active("experimental-policy") is experimental


def test_registry_rejects_activation_when_another_version_is_active() -> None:
    registry = RoutingPolicyRegistry()

    active = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    approved = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(active)
    registry.register(approved)

    with pytest.raises(
        ValueError,
        match="already has an active version",
    ):
        registry.transition(
            identifier=approved.identifier,
            target_status=RoutingPolicyLifecycleStatus.ACTIVE,
        )

    assert (
        registry.get(approved.identifier).status
        is RoutingPolicyLifecycleStatus.APPROVED
    )

    assert registry.get_active("production-policy") is active


def test_registry_allows_new_activation_after_previous_retired() -> None:
    registry = RoutingPolicyRegistry()

    old = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    new = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(old)
    registry.register(new)

    retire_result = registry.transition(
        identifier=old.identifier,
        target_status=RoutingPolicyLifecycleStatus.RETIRED,
    )

    assert retire_result.allowed is True

    activate_result = registry.transition(
        identifier=new.identifier,
        target_status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    assert activate_result.allowed is True

    active = registry.get_active("production-policy")

    assert active is not None
    assert active.identifier == ("production-policy@1.1.0")


def test_failed_activation_does_not_mutate_registry_state() -> None:
    registry = RoutingPolicyRegistry()

    active = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    approved = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(active)
    registry.register(approved)

    before = registry.list_all()

    with pytest.raises(
        ValueError,
        match="already has an active version",
    ):
        registry.transition(
            identifier=approved.identifier,
            target_status=RoutingPolicyLifecycleStatus.ACTIVE,
        )

    assert registry.list_all() == before


def test_activate_approved_policy_without_existing_active() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(candidate)

    result = registry.activate(candidate.identifier)

    assert result.activated_policy.status == RoutingPolicyLifecycleStatus.ACTIVE

    assert result.retired_policy is None

    assert registry.get_active("production-policy") == result.activated_policy


def test_activate_retires_existing_active_policy() -> None:
    registry = RoutingPolicyRegistry()

    current = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(current)
    registry.register(candidate)

    result = registry.activate(candidate.identifier)

    assert result.retired_policy is not None

    assert result.retired_policy.identifier == current.identifier

    assert (
        registry.get(current.identifier).status == RoutingPolicyLifecycleStatus.RETIRED
    )

    assert (
        registry.get(candidate.identifier).status == RoutingPolicyLifecycleStatus.ACTIVE
    )


def test_activate_returns_new_active_policy() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_governed_policy(
        version="2.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(candidate)

    result = registry.activate(candidate.identifier)

    assert result.activated_policy.identifier == (candidate.identifier)

    assert result.activated_policy.versioned_policy is candidate.versioned_policy


def test_activate_rejects_candidate_status() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    registry.register(candidate)

    before = registry.list_all()

    with pytest.raises(
        ValueError,
        match="cannot be activated",
    ):
        registry.activate(candidate.identifier)

    assert registry.list_all() == before


def test_activate_rejects_draft_status() -> None:
    registry = RoutingPolicyRegistry()

    draft = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.DRAFT,
    )

    registry.register(draft)

    with pytest.raises(
        ValueError,
        match="cannot be activated",
    ):
        registry.activate(draft.identifier)

    assert registry.get(draft.identifier).status == RoutingPolicyLifecycleStatus.DRAFT


def test_activate_rejects_retired_policy() -> None:
    registry = RoutingPolicyRegistry()

    retired = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.RETIRED,
    )

    registry.register(retired)

    with pytest.raises(
        ValueError,
        match="cannot be activated",
    ):
        registry.activate(retired.identifier)


def test_activate_missing_policy_raises() -> None:
    registry = RoutingPolicyRegistry()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        registry.activate("missing-policy@1.0.0")


def test_activation_leaves_exactly_one_active_version() -> None:
    registry = RoutingPolicyRegistry()

    current = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_governed_policy(
        name="production-policy",
        version="2.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(current)
    registry.register(candidate)

    registry.activate(candidate.identifier)

    active = registry.find_by_name_and_status(
        policy_name="production-policy",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    assert len(active) == 1
    assert active[0].identifier == candidate.identifier


def test_activation_history_starts_empty() -> None:
    registry = RoutingPolicyRegistry()

    assert registry.activation_history() == ()


def test_activation_records_audit_event() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(candidate)

    registry.activate(candidate.identifier)

    history = registry.activation_history()

    assert len(history) == 1

    event = history[0]

    assert event.policy_name == "production-policy"

    assert event.activated_policy_identifier == ("production-policy@1.0.0")

    assert event.retired_policy_identifier is None


def test_activation_history_records_replaced_policy() -> None:
    registry = RoutingPolicyRegistry()

    current = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.ACTIVE,
    )

    candidate = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(current)
    registry.register(candidate)

    registry.activate(candidate.identifier)

    event = registry.activation_history()[0]

    assert event.activated_policy_identifier == ("production-policy@1.1.0")

    assert event.retired_policy_identifier == ("production-policy@1.0.0")


def test_activation_history_preserves_activation_order() -> None:
    registry = RoutingPolicyRegistry()

    first = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    second = create_governed_policy(
        name="production-policy",
        version="1.1.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(first)
    registry.register(second)

    registry.activate(first.identifier)

    registry.activate(second.identifier)

    history = registry.activation_history()

    assert tuple(event.activated_policy_identifier for event in history) == (
        "production-policy@1.0.0",
        "production-policy@1.1.0",
    )


def test_activation_history_can_filter_by_policy_name() -> None:
    registry = RoutingPolicyRegistry()

    production = create_governed_policy(
        name="production-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    experimental = create_governed_policy(
        name="experimental-policy",
        version="1.0.0",
        status=RoutingPolicyLifecycleStatus.APPROVED,
    )

    registry.register(production)
    registry.register(experimental)

    registry.activate(production.identifier)

    registry.activate(experimental.identifier)

    production_history = registry.activation_history_for("production-policy")

    assert len(production_history) == 1

    assert (
        production_history[0].activated_policy_identifier == "production-policy@1.0.0"
    )


def test_failed_activation_does_not_create_audit_event() -> None:
    registry = RoutingPolicyRegistry()

    candidate = create_governed_policy(
        status=RoutingPolicyLifecycleStatus.CANDIDATE,
    )

    registry.register(candidate)

    with pytest.raises(
        ValueError,
        match="cannot be activated",
    ):
        registry.activate(candidate.identifier)

    assert registry.activation_history() == ()
