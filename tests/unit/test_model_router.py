import pytest

from agent_platform.llm.errors import (
    LLMModelConstraintViolationError,
    LLMModelDisabledError,
    LLMModelNotFoundError,
    LLMModelWorkloadNotSupportedError,
)
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_registry import ModelRegistry
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.model_tier import ModelCostTier
from agent_platform.llm.routing_constraints import RoutingConstraints
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


def test_model_router_filters_candidates_by_cost_constraint() -> None:
    expensive = ModelDefinition(
        name="expensive",
        provider="openai",
        provider_model="expensive-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
        cost_tier=ModelCostTier.HIGH,
    )

    cheap = ModelDefinition(
        name="cheap",
        provider="openai",
        provider_model="cheap-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
        cost_tier=ModelCostTier.LOW,
    )

    router = ModelRouter(
        models={
            "expensive": expensive,
            "cheap": cheap,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: (
                    "expensive",
                    "cheap",
                ),
            }
        ),
    )

    result = router.route_candidates(
        LLMWorkload.CLASSIFICATION,
        constraints=RoutingConstraints(
            max_cost_tier=ModelCostTier.LOW,
        ),
    )

    assert result == (cheap,)


def test_model_router_filters_candidates_by_provider_constraint() -> None:
    openai_model = ModelDefinition(
        name="openai_model",
        provider="openai",
        provider_model="openai-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    other_model = ModelDefinition(
        name="other_model",
        provider="other",
        provider_model="other-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    router = ModelRouter(
        models={
            "openai_model": openai_model,
            "other_model": other_model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: (
                    "openai_model",
                    "other_model",
                ),
            }
        ),
    )

    result = router.route_candidates(
        LLMWorkload.CLASSIFICATION,
        constraints=RoutingConstraints(
            allowed_providers=frozenset(
                {
                    "openai",
                }
            ),
        ),
    )

    assert result == (openai_model,)


def test_model_router_raises_when_constraints_reject_all_candidates() -> None:
    model = ModelDefinition(
        name="expensive",
        provider="openai",
        provider_model="expensive-model",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
        cost_tier=ModelCostTier.HIGH,
    )

    router = ModelRouter(
        models={
            "expensive": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: "expensive",
            }
        ),
    )

    with pytest.raises(
        LLMModelConstraintViolationError,
        match="classification",
    ):
        router.route_candidates(
            LLMWorkload.CLASSIFICATION,
            constraints=RoutingConstraints(
                max_cost_tier=ModelCostTier.LOW,
            ),
        )


def test_model_router_can_use_model_registry() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    registry = ModelRegistry(
        models={
            "fast_general": model,
        }
    )

    router = ModelRouter(
        registry=registry,
        policy=ModelPolicy(
            assignments={
                LLMWorkload.CLASSIFICATION: "fast_general",
            }
        ),
    )

    assert router.route(LLMWorkload.CLASSIFICATION) is model


def test_model_router_still_accepts_models_dictionary() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
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

    assert router.route(LLMWorkload.CLASSIFICATION) is model


def test_model_router_filters_by_required_capability() -> None:
    text_only = ModelDefinition(
        name="text_only",
        provider="openai",
        provider_model="text-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
            }
        ),
    )

    tool_capable = ModelDefinition(
        name="tool_capable",
        provider="openai",
        provider_model="tool-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.TOOL_CALLING,
            }
        ),
    )

    router = ModelRouter(
        models={
            "text_only": text_only,
            "tool_capable": tool_capable,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "text_only",
                    "tool_capable",
                ),
            }
        ),
    )

    candidates = router.route_candidates(
        LLMWorkload.GENERAL,
        required_capabilities=frozenset(
            {
                ModelCapability.TOOL_CALLING,
            }
        ),
    )

    assert candidates == (tool_capable,)


def test_model_router_requires_all_requested_capabilities() -> None:
    partial = ModelDefinition(
        name="partial",
        provider="openai",
        provider_model="partial-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
            }
        ),
    )

    complete = ModelDefinition(
        name="complete",
        provider="openai",
        provider_model="complete-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.TOOL_CALLING,
            }
        ),
    )

    router = ModelRouter(
        models={
            "partial": partial,
            "complete": complete,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "partial",
                    "complete",
                ),
            }
        ),
    )

    candidates = router.route_candidates(
        LLMWorkload.GENERAL,
        required_capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.TOOL_CALLING,
            }
        ),
    )

    assert candidates == (complete,)


def test_model_router_raises_when_capabilities_reject_all_candidates() -> None:
    model = ModelDefinition(
        name="text_only",
        provider="openai",
        provider_model="text-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
            }
        ),
    )

    router = ModelRouter(
        models={
            "text_only": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "text_only",
            }
        ),
    )

    with pytest.raises(
        LLMModelConstraintViolationError,
        match="general",
    ):
        router.route_candidates(
            LLMWorkload.GENERAL,
            required_capabilities=frozenset(
                {
                    ModelCapability.VISION,
                }
            ),
        )
