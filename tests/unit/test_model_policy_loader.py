from agent_platform.config import Settings
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.model_policy_loader import load_model_policy
from agent_platform.llm.workload import LLMWorkload


def test_load_model_policy_from_settings() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_model_policy=ModelPolicyConfig(
            assignments={
                LLMWorkload.CLASSIFICATION: (
                    "classification_primary",
                    "classification_backup",
                ),
                LLMWorkload.REASONING: "reasoning_model",
            }
        ),
    )

    policy = load_model_policy(settings)

    assert policy.models_for(LLMWorkload.CLASSIFICATION) == (
        "classification_primary",
        "classification_backup",
    )

    assert policy.model_for(LLMWorkload.REASONING) == "reasoning_model"


def test_load_model_policy_with_empty_configuration() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    policy = load_model_policy(settings)

    assert policy.assignments == {}
