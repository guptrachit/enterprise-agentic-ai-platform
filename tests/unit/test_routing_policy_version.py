import pytest

from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)


def test_policy_version_preserves_components() -> None:
    version = RoutingPolicyVersion(
        major=1,
        minor=4,
        patch=2,
    )

    assert version.major == 1
    assert version.minor == 4
    assert version.patch == 2
    assert version.value == "1.4.2"
    assert str(version) == "1.4.2"


def test_policy_version_defaults_patch_to_zero() -> None:
    version = RoutingPolicyVersion(
        major=2,
        minor=1,
    )

    assert version.patch == 0
    assert version.value == "2.1.0"


def test_policy_version_parse_three_components() -> None:
    version = RoutingPolicyVersion.parse("3.5.7")

    assert version == RoutingPolicyVersion(
        major=3,
        minor=5,
        patch=7,
    )


def test_policy_version_parse_two_components() -> None:
    version = RoutingPolicyVersion.parse("3.5")

    assert version == RoutingPolicyVersion(
        major=3,
        minor=5,
        patch=0,
    )


def test_policy_versions_are_orderable() -> None:
    older = RoutingPolicyVersion(
        major=1,
        minor=2,
        patch=0,
    )

    newer = RoutingPolicyVersion(
        major=1,
        minor=3,
        patch=0,
    )

    assert older < newer
    assert newer > older


@pytest.mark.parametrize(
    "value",
    (
        "1",
        "1.2.3.4",
        "",
    ),
)
def test_policy_version_rejects_invalid_shape(
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="major.minor",
    ):
        RoutingPolicyVersion.parse(value)


@pytest.mark.parametrize(
    "value",
    (
        "one.two",
        "1.two.3",
    ),
)
def test_policy_version_rejects_non_integer_components(
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be integers",
    ):
        RoutingPolicyVersion.parse(value)


@pytest.mark.parametrize(
    (
        "major",
        "minor",
        "patch",
    ),
    (
        (-1, 0, 0),
        (1, -1, 0),
        (1, 0, -1),
    ),
)
def test_policy_version_rejects_negative_components(
    major: int,
    minor: int,
    patch: int,
) -> None:
    with pytest.raises(ValueError):
        RoutingPolicyVersion(
            major=major,
            minor=minor,
            patch=patch,
        )
