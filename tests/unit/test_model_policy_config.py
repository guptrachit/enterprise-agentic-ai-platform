from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.workload import LLMWorkload


def test_model_policy_config_parses_single_model_assignment() -> None:
    config = ModelPolicyConfig(
        assignments={
            LLMWorkload.CLASSIFICATION: "fast_general",
        }
    )

    assert config.assignments[LLMWorkload.CLASSIFICATION] == "fast_general"


def test_model_policy_config_supports_fallback_chain() -> None:
    config = ModelPolicyConfig(
        assignments={
            LLMWorkload.CLASSIFICATION: (
                "classification_primary",
                "classification_backup",
            ),
        }
    )

    assert config.assignments[LLMWorkload.CLASSIFICATION] == (
        "classification_primary",
        "classification_backup",
    )
