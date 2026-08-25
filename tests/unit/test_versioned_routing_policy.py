from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.routing_policy_metadata import (
    RoutingPolicyMetadata,
)
from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)
from agent_platform.llm.versioned_routing_policy import (
    VersionedRoutingPolicy,
)
from agent_platform.llm.workload import LLMWorkload


def test_versioned_policy_preserves_policy_and_metadata() -> None:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: (
                "primary",
                "backup",
            ),
        }
    )

    metadata = RoutingPolicyMetadata(
        name="production-routing-policy",
        version=RoutingPolicyVersion(
            major=1,
            minor=4,
            patch=0,
        ),
        description="Production routing policy",
    )

    versioned_policy = VersionedRoutingPolicy(
        policy=policy,
        metadata=metadata,
    )

    assert versioned_policy.policy is policy
    assert versioned_policy.metadata is metadata


def test_versioned_policy_identifier() -> None:
    versioned_policy = VersionedRoutingPolicy(
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
        metadata=RoutingPolicyMetadata(
            name="candidate-policy",
            version=RoutingPolicyVersion(
                major=2,
                minor=1,
                patch=3,
            ),
        ),
    )

    assert versioned_policy.identifier == ("candidate-policy@2.1.3")


def test_versioned_policy_preserves_model_assignments() -> None:
    versioned_policy = VersionedRoutingPolicy(
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "primary",
                    "backup",
                ),
                LLMWorkload.CLASSIFICATION: (
                    "classification_primary",
                    "classification_backup",
                ),
            }
        ),
        metadata=RoutingPolicyMetadata(
            name="multi-workload-policy",
            version=RoutingPolicyVersion(
                major=3,
                minor=0,
            ),
        ),
    )

    assert versioned_policy.policy.models_for(LLMWorkload.GENERAL) == (
        "primary",
        "backup",
    )

    assert versioned_policy.policy.models_for(LLMWorkload.CLASSIFICATION) == (
        "classification_primary",
        "classification_backup",
    )
