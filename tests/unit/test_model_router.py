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
from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.model_registry import ModelRegistry
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
    provider: str = "openai",
    workloads: frozenset[LLMWorkload] | None = None,
    enabled: bool = True,
    cost_tier: ModelCostTier = ModelCostTier.MEDIUM,
    latency_tier: ModelLatencyTier = ModelLatencyTier.STANDARD,
    capabilities: frozenset[ModelCapability] | None = None,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider=provider,
        provider_model=f"{name}-provider-model",
        workloads=workloads
        or frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        enabled=enabled,
        cost_tier=cost_tier,
        latency_tier=latency_tier,
        capabilities=capabilities,
    )


def test_model_router_returns_expected_model() -> None:
    model = create_model(
        name="fast_general",
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

    assert router.route(LLMWorkload.CLASSIFICATION) is model


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
    model = create_model(
        name="disabled_model",
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
    model = create_model(
        name="general_model",
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
    primary = create_model(
        name="primary",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    disabled_backup = create_model(
        name="disabled_backup",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
        enabled=False,
    )

    backup = create_model(
        name="backup",
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
    expensive = create_model(
        name="expensive",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
        cost_tier=ModelCostTier.HIGH,
    )

    cheap = create_model(
        name="cheap",
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
    openai_model = create_model(
        name="openai_model",
        provider="openai",
        workloads=frozenset(
            {
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    other_model = create_model(
        name="other_model",
        provider="other",
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
    model = create_model(
        name="expensive",
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
    model = create_model(
        name="fast_general",
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
    model = create_model(
        name="fast_general",
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
    text_only = create_model(
        name="text_only",
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
            }
        ),
    )

    tool_capable = create_model(
        name="tool_capable",
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
    partial = create_model(
        name="partial",
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
            }
        ),
    )

    complete = create_model(
        name="complete",
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
    model = create_model(
        name="text_only",
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


def test_model_router_ranks_eligible_candidates_by_lower_cost() -> None:
    expensive = create_model(
        name="expensive",
        cost_tier=ModelCostTier.HIGH,
    )

    cheap = create_model(
        name="cheap",
        cost_tier=ModelCostTier.LOW,
    )

    router = ModelRouter(
        models={
            "expensive": expensive,
            "cheap": cheap,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "expensive",
                    "cheap",
                ),
            }
        ),
    )

    candidates = router.route_candidates(
        LLMWorkload.GENERAL,
        preference=ModelPreference(
            prefer_lower_cost=True,
        ),
    )

    assert candidates == (
        cheap,
        expensive,
    )


def test_model_router_ranks_eligible_candidates_by_lower_latency() -> None:
    slow = create_model(
        name="slow",
        latency_tier=ModelLatencyTier.SLOW,
    )

    fast = create_model(
        name="fast",
        latency_tier=ModelLatencyTier.FAST,
    )

    router = ModelRouter(
        models={
            "slow": slow,
            "fast": fast,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "slow",
                    "fast",
                ),
            }
        ),
    )

    candidates = router.route_candidates(
        LLMWorkload.GENERAL,
        preference=ModelPreference(
            prefer_lower_latency=True,
        ),
    )

    assert candidates == (
        fast,
        slow,
    )


def test_model_router_ranks_by_preferred_provider() -> None:
    second_provider = create_model(
        name="second_provider",
        provider="anthropic",
    )

    preferred = create_model(
        name="preferred",
        provider="openai",
    )

    router = ModelRouter(
        models={
            "second_provider": second_provider,
            "preferred": preferred,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "second_provider",
                    "preferred",
                ),
            }
        ),
    )

    candidates = router.route_candidates(
        LLMWorkload.GENERAL,
        preference=ModelPreference(
            preferred_providers=(
                "openai",
                "anthropic",
            ),
        ),
    )

    assert candidates == (
        preferred,
        second_provider,
    )


def test_model_router_preserves_policy_order_without_preference() -> None:
    first = create_model(
        name="first",
        cost_tier=ModelCostTier.HIGH,
    )

    second = create_model(
        name="second",
        cost_tier=ModelCostTier.LOW,
    )

    router = ModelRouter(
        models={
            "first": first,
            "second": second,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "first",
                    "second",
                ),
            }
        ),
    )

    candidates = router.route_candidates(LLMWorkload.GENERAL)

    assert candidates == (
        first,
        second,
    )
