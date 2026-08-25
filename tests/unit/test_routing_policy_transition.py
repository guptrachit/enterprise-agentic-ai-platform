import pytest

from agent_platform.llm.routing_policy_transition import (
    RoutingPolicyTransitionStatus,
    evaluate_routing_policy_transition,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)


def version(
    value: str,
) -> RoutingPolicyVersion:
    return RoutingPolicyVersion.parse(value)


@pytest.mark.parametrize(
    (
        "baseline",
        "candidate",
    ),
    (
        ("1.0.0", "1.0.1"),
        ("1.0.9", "1.1.0"),
        ("1.9.9", "2.0.0"),
        ("0.1.0", "1.0.0"),
    ),
)
def test_newer_policy_version_is_allowed(
    baseline: str,
    candidate: str,
) -> None:
    result = evaluate_routing_policy_transition(
        baseline_version=version(baseline),
        candidate_version=version(candidate),
    )

    assert result.status is RoutingPolicyTransitionStatus.ALLOWED
    assert result.allowed is True

    assert result.baseline_version == version(baseline)

    assert result.candidate_version == version(candidate)


def test_same_policy_version_is_rejected() -> None:
    result = evaluate_routing_policy_transition(
        baseline_version=version("1.5.0"),
        candidate_version=version("1.5.0"),
    )

    assert result.status is RoutingPolicyTransitionStatus.REJECTED
    assert result.allowed is False

    assert result.reason == (
        "Candidate routing policy version must be newer than the baseline version."
    )


@pytest.mark.parametrize(
    (
        "baseline",
        "candidate",
    ),
    (
        ("1.5.0", "1.4.9"),
        ("2.0.0", "1.9.9"),
        ("1.1.0", "1.0.9"),
    ),
)
def test_older_policy_version_is_rejected(
    baseline: str,
    candidate: str,
) -> None:
    result = evaluate_routing_policy_transition(
        baseline_version=version(baseline),
        candidate_version=version(candidate),
    )

    assert result.status is RoutingPolicyTransitionStatus.REJECTED
    assert result.allowed is False

    assert result.reason == (
        "Candidate routing policy version is older than the baseline version."
    )


def test_transition_result_preserves_versions() -> None:
    baseline = version("3.2.1")
    candidate = version("3.3.0")

    result = evaluate_routing_policy_transition(
        baseline_version=baseline,
        candidate_version=candidate,
    )

    assert result.baseline_version is baseline
    assert result.candidate_version is candidate
