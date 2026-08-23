import pytest

from agent_platform.llm.errors import (
    LLMModelDisabledError,
    LLMModelNotFoundError,
    LLMModelWorkloadNotSupportedError,
)
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.workload import LLMWorkload


def test_model_router_returns_expected_model() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = ModelRouter(
        models={
            "fast_general": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: "fast_general",
            }
        ),
    )

    result = router.route(LLMWorkload.CLASSIFICATION)

    assert result is model


def test_model_router_raises_when_model_not_found() -> None:
    router = ModelRouter(
        models={},
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: "missing_model",
            }
        ),
    )

    with pytest.raises(
        LLMModelNotFoundError,
        match="missing_model",
    ):
        router.route(LLMWorkload.CLASSIFICATION)


def test_model_router_rejects_disabled_model() -> None:
    model = ModelDefinition(
        name="disabled_model",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
        enabled=False,
    )

    router = ModelRouter(
        models={
            "disabled_model": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: "disabled_model",
            }
        ),
    )

    with pytest.raises(
        LLMModelDisabledError,
        match="disabled_model",
    ):
        router.route(LLMWorkload.CLASSIFICATION)


def test_model_router_rejects_unsupported_workload() -> None:
    model = ModelDefinition(
        name="general_model",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )

    router = ModelRouter(
        models={
            "general_model": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.REASONING: "general_model",
            }
        ),
    )

    with pytest.raises(
        LLMModelWorkloadNotSupportedError,
        match="reasoning",
    ):
        router.route(LLMWorkload.REASONING)


def test_model_router_returns_ordered_valid_candidates() -> None:
    primary = ModelDefinition(
        name="primary",
        provider="openai",
        provider_model="primary-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    disabled_backup = ModelDefinition(
        name="disabled_backup",
        provider="openai",
        provider_model="disabled-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
        enabled=False,
    )

    backup = ModelDefinition(
        name="backup",
        provider="openai",
        provider_model="backup-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = ModelRouter(
        models={
            "primary": primary,
            "disabled_backup": disabled_backup,
            "backup": backup,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: (
                    "primary",
                    "disabled_backup",
                    "backup",
                ),
            }
        ),
    )

    assert router.route_candidates(LLMWorkload.CLASSIFICATION) == (
        primary,
        backup,
    )
