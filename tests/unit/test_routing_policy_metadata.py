import pytest

from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)


def test_policy_metadata_preserves_values() -> None:
    version = RoutingPolicyVersion(
        major=1,
        minor=4,
        patch=0,
    )

    metadata = RoutingPolicyMetadata(
        name="production-routing-policy",
        version=version,
        description="Lower-cost preference rollout",
    )

    assert metadata.name == "production-routing-policy"
    assert metadata.version is version

    assert metadata.description == ("Lower-cost preference rollout")


def test_policy_metadata_description_is_optional() -> None:
    metadata = RoutingPolicyMetadata(
        name="default-policy",
        version=RoutingPolicyVersion(
            major=1,
            minor=0,
        ),
    )

    assert metadata.description is None


def test_policy_metadata_identifier() -> None:
    metadata = RoutingPolicyMetadata(
        name="production-routing-policy",
        version=RoutingPolicyVersion(
            major=2,
            minor=3,
            patch=1,
        ),
    )

    assert metadata.identifier == ("production-routing-policy@2.3.1")


@pytest.mark.parametrize(
    "name",
    (
        "",
        " ",
        "   ",
    ),
)
def test_policy_metadata_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        RoutingPolicyMetadata(
            name=name,
            version=RoutingPolicyVersion(
                major=1,
                minor=0,
            ),
        )
