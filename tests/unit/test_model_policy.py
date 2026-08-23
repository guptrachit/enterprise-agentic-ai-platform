import pytest

from agent_platform.llm.errors import LLMModelPolicyNotFoundError
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.workload import LLMWorkload


def test_model_policy_returns_assigned_model() -> None:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.CLASSIFICATION: "fast_general",
            LLMWorkload.REASONING: "reasoning_model",
        }
    )

    assert policy.model_for(LLMWorkload.CLASSIFICATION) == "fast_general"


def test_model_policy_supports_multiple_workloads() -> None:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "fast_general",
            LLMWorkload.CLASSIFICATION: "fast_general",
            LLMWorkload.EXTRACTION: "structured_model",
            LLMWorkload.REASONING: "reasoning_model",
        }
    )

    assert policy.model_for(LLMWorkload.GENERAL) == "fast_general"

    assert policy.model_for(LLMWorkload.EXTRACTION) == "structured_model"

    assert policy.model_for(LLMWorkload.REASONING) == "reasoning_model"


def test_model_policy_raises_when_workload_not_configured() -> None:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "fast_general",
        }
    )

    with pytest.raises(
        LLMModelPolicyNotFoundError,
        match="reasoning",
    ):
        policy.model_for(LLMWorkload.REASONING)


def test_model_policy_returns_ordered_fallback_models() -> None:
    policy = ModelPolicy(
        assignments={
            LLMWorkload.CLASSIFICATION: (
                "classification_primary",
                "classification_backup",
            ),
        }
    )

    assert policy.models_for(LLMWorkload.CLASSIFICATION) == (
        "classification_primary",
        "classification_backup",
    )

    assert policy.model_for(LLMWorkload.CLASSIFICATION) == "classification_primary"
