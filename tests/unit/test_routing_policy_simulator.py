import pytest

from agent_platform.llm.errors import LLMModelConstraintViolationError
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.model_tier import (
    ModelCostTier,
    ModelLatencyTier,
)
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.routing_policy_metadata import RoutingPolicyMetadata
from agent_platform.llm.routing_policy_simulator import (
    RoutingPolicySimulator,
)
from agent_platform.llm.routing_policy_version import RoutingPolicyVersion
from agent_platform.llm.routing_simulation import RoutingSimulationCase
from agent_platform.llm.versioned_routing_policy import VersionedRoutingPolicy
from agent_platform.llm.workload import LLMWorkload


def create_model(
    *,
    name: str,
    provider: str = "openai",
    workload: LLMWorkload = LLMWorkload.GENERAL,
    cost_tier: ModelCostTier = ModelCostTier.MEDIUM,
    latency_tier: ModelLatencyTier = ModelLatencyTier.STANDARD,
    capabilities: frozenset[ModelCapability] | None = None,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider=provider,
        provider_model=f"{name}-provider-model",
        workloads=frozenset(
            {
                workload,
            }
        ),
        cost_tier=cost_tier,
        latency_tier=latency_tier,
        capabilities=capabilities,
    )


def test_simulator_returns_routing_decision() -> None:
    model = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    case = RoutingSimulationCase(
        name="general-case",
        workload=LLMWorkload.GENERAL,
    )

    result = simulator.simulate(case)

    assert result.case is case
    assert result.selected_model is model
    assert result.decision.selected_model is model


def test_simulator_does_not_execute_llm_provider() -> None:
    model = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="offline-case",
            workload=LLMWorkload.GENERAL,
        )
    )

    assert result.selected_model.name == "primary"


def test_simulate_many_preserves_case_order() -> None:
    general_model = create_model(
        name="general_model",
        workload=LLMWorkload.GENERAL,
    )

    classification_model = create_model(
        name="classification_model",
        workload=LLMWorkload.CLASSIFICATION,
    )

    router = ModelRouter(
        models={
            "general_model": general_model,
            "classification_model": classification_model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "general_model",
                LLMWorkload.CLASSIFICATION: "classification_model",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    cases = (
        RoutingSimulationCase(
            name="general-case",
            workload=LLMWorkload.GENERAL,
        ),
        RoutingSimulationCase(
            name="classification-case",
            workload=LLMWorkload.CLASSIFICATION,
        ),
    )

    results = simulator.simulate_many(cases)

    assert tuple(result.case.name for result in results) == (
        "general-case",
        "classification-case",
    )

    assert tuple(result.selected_model.name for result in results) == (
        "general_model",
        "classification_model",
    )


def test_simulation_respects_provider_constraint() -> None:
    openai_model = create_model(
        name="openai_model",
        provider="openai",
    )

    other_model = create_model(
        name="other_model",
        provider="anthropic",
    )

    router = ModelRouter(
        models={
            "openai_model": openai_model,
            "other_model": other_model,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "other_model",
                    "openai_model",
                ),
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="openai-only",
            workload=LLMWorkload.GENERAL,
            constraints=RoutingConstraints(
                allowed_providers=frozenset(
                    {
                        "openai",
                    }
                ),
            ),
        )
    )

    assert result.selected_model is openai_model


def test_simulation_respects_cost_constraint() -> None:
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

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="low-cost-only",
            workload=LLMWorkload.GENERAL,
            constraints=RoutingConstraints(
                max_cost_tier=ModelCostTier.LOW,
            ),
        )
    )

    assert result.selected_model is cheap


def test_simulation_raises_when_constraints_reject_all_models() -> None:
    expensive = create_model(
        name="expensive",
        cost_tier=ModelCostTier.HIGH,
    )

    router = ModelRouter(
        models={
            "expensive": expensive,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "expensive",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    with pytest.raises(LLMModelConstraintViolationError):
        simulator.simulate(
            RoutingSimulationCase(
                name="impossible-cost-policy",
                workload=LLMWorkload.GENERAL,
                constraints=RoutingConstraints(
                    max_cost_tier=ModelCostTier.LOW,
                ),
            )
        )


def test_simulation_respects_required_capability() -> None:
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

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="tool-required",
            workload=LLMWorkload.GENERAL,
            required_capabilities=frozenset(
                {
                    ModelCapability.TOOL_CALLING,
                }
            ),
        )
    )

    assert result.selected_model is tool_capable


def test_simulation_prefers_lower_cost() -> None:
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

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="prefer-cheap",
            workload=LLMWorkload.GENERAL,
            preference=ModelPreference(
                prefer_lower_cost=True,
            ),
        )
    )

    assert result.selected_model is cheap


def test_simulation_prefers_lower_latency() -> None:
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

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="prefer-fast",
            workload=LLMWorkload.GENERAL,
            preference=ModelPreference(
                prefer_lower_latency=True,
            ),
        )
    )

    assert result.selected_model is fast


def test_simulation_prefers_provider_order() -> None:
    anthropic = create_model(
        name="anthropic_model",
        provider="anthropic",
    )

    openai = create_model(
        name="openai_model",
        provider="openai",
    )

    router = ModelRouter(
        models={
            "anthropic_model": anthropic,
            "openai_model": openai,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "anthropic_model",
                    "openai_model",
                ),
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="prefer-openai",
            workload=LLMWorkload.GENERAL,
            preference=ModelPreference(
                preferred_providers=(
                    "openai",
                    "anthropic",
                ),
            ),
        )
    )

    assert result.selected_model is openai


def test_simulation_combines_constraint_and_preference() -> None:
    expensive_openai = create_model(
        name="expensive_openai",
        provider="openai",
        cost_tier=ModelCostTier.HIGH,
    )

    cheap_openai = create_model(
        name="cheap_openai",
        provider="openai",
        cost_tier=ModelCostTier.LOW,
    )

    cheap_other = create_model(
        name="cheap_other",
        provider="anthropic",
        cost_tier=ModelCostTier.LOW,
    )

    router = ModelRouter(
        models={
            "expensive_openai": expensive_openai,
            "cheap_openai": cheap_openai,
            "cheap_other": cheap_other,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: (
                    "expensive_openai",
                    "cheap_other",
                    "cheap_openai",
                ),
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="openai-and-cheap",
            workload=LLMWorkload.GENERAL,
            constraints=RoutingConstraints(
                allowed_providers=frozenset(
                    {
                        "openai",
                    }
                ),
            ),
            preference=ModelPreference(
                prefer_lower_cost=True,
            ),
        )
    )

    assert result.selected_model is cheap_openai


def test_simulator_validates_expected_selection() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="expected-primary",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="primary",
        )
    )

    assert result.matches_expectation is True


def test_simulator_detects_expectation_failure() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    result = simulator.simulate(
        RoutingSimulationCase(
            name="expected-backup",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="backup",
        )
    )

    assert result.matches_expectation is False


def test_expectation_failures_returns_only_failed_cases() -> None:
    primary = create_model(
        name="primary",
    )

    classification = create_model(
        name="classification",
        workload=LLMWorkload.CLASSIFICATION,
    )

    router = ModelRouter(
        models={
            "primary": primary,
            "classification": classification,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
                LLMWorkload.CLASSIFICATION: "classification",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    cases = (
        RoutingSimulationCase(
            name="general-good",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="primary",
        ),
        RoutingSimulationCase(
            name="classification-bad",
            workload=LLMWorkload.CLASSIFICATION,
            expected_selected_model="wrong_model",
        ),
        RoutingSimulationCase(
            name="general-no-expectation",
            workload=LLMWorkload.GENERAL,
        ),
    )

    failures = simulator.expectation_failures(cases)

    assert len(failures) == 1

    assert failures[0].case.name == "classification-bad"

    assert failures[0].selected_model.name == "classification"

    assert failures[0].matches_expectation is False


def test_simulation_summary_all_expectations_pass() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    cases = (
        RoutingSimulationCase(
            name="case-1",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="primary",
        ),
        RoutingSimulationCase(
            name="case-2",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="primary",
        ),
    )

    summary = simulator.summarize(cases)

    assert summary.total_cases == 2
    assert summary.cases_with_expectations == 2
    assert summary.passed_expectations == 2
    assert summary.failed_expectations == 0
    assert summary.failed_case_names == ()
    assert summary.expectation_pass_rate == 1.0
    assert summary.passed is True


def test_simulation_summary_detects_failed_expectations() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    cases = (
        RoutingSimulationCase(
            name="good-case",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="primary",
        ),
        RoutingSimulationCase(
            name="bad-case",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="backup",
        ),
    )

    summary = simulator.summarize(cases)

    assert summary.total_cases == 2
    assert summary.cases_with_expectations == 2
    assert summary.passed_expectations == 1
    assert summary.failed_expectations == 1

    assert summary.failed_case_names == ("bad-case",)

    assert summary.expectation_pass_rate == 0.5
    assert summary.passed is False


def test_simulation_summary_ignores_cases_without_expectations() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    cases = (
        RoutingSimulationCase(
            name="evaluated-case",
            workload=LLMWorkload.GENERAL,
            expected_selected_model="primary",
        ),
        RoutingSimulationCase(
            name="informational-case",
            workload=LLMWorkload.GENERAL,
        ),
    )

    summary = simulator.summarize(cases)

    assert summary.total_cases == 2
    assert summary.cases_with_expectations == 1
    assert summary.passed_expectations == 1
    assert summary.failed_expectations == 0
    assert summary.expectation_pass_rate == 1.0
    assert summary.passed is True


def test_simulation_summary_with_no_expectations_is_passing() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    cases = (
        RoutingSimulationCase(
            name="informational-case",
            workload=LLMWorkload.GENERAL,
        ),
    )

    summary = simulator.summarize(cases)

    assert summary.total_cases == 1
    assert summary.cases_with_expectations == 0
    assert summary.passed_expectations == 0
    assert summary.failed_expectations == 0
    assert summary.failed_case_names == ()
    assert summary.expectation_pass_rate == 1.0
    assert summary.passed is True


def test_empty_simulation_summary_is_passing() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    summary = simulator.summarize(())

    assert summary.total_cases == 0
    assert summary.cases_with_expectations == 0
    assert summary.passed_expectations == 0
    assert summary.failed_expectations == 0
    assert summary.failed_case_names == ()
    assert summary.expectation_pass_rate == 1.0
    assert summary.passed is True


def test_simulator_exposes_policy_identifier() -> None:
    primary = create_model(
        name="primary",
    )

    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "primary",
        }
    )

    versioned_policy = VersionedRoutingPolicy(
        policy=policy,
        metadata=RoutingPolicyMetadata(
            name="candidate-policy",
            version=RoutingPolicyVersion(
                major=1,
                minor=4,
                patch=0,
            ),
        ),
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=policy,
    )

    simulator = RoutingPolicySimulator(
        router,
        versioned_policy=versioned_policy,
    )

    assert simulator.policy_identifier == ("candidate-policy@1.4.0")


def test_simulator_without_versioned_policy_has_no_identifier() -> None:
    primary = create_model(
        name="primary",
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=ModelPolicy(
            assignments={
                LLMWorkload.GENERAL: "primary",
            }
        ),
    )

    simulator = RoutingPolicySimulator(router)

    assert simulator.policy_identifier is None


def test_simulation_summary_includes_policy_identifier() -> None:
    primary = create_model(
        name="primary",
    )

    policy = ModelPolicy(
        assignments={
            LLMWorkload.GENERAL: "primary",
        }
    )

    versioned_policy = VersionedRoutingPolicy(
        policy=policy,
        metadata=RoutingPolicyMetadata(
            name="production-routing-policy",
            version=RoutingPolicyVersion(
                major=2,
                minor=1,
                patch=3,
            ),
        ),
    )

    router = ModelRouter(
        models={
            "primary": primary,
        },
        policy=policy,
    )

    simulator = RoutingPolicySimulator(
        router,
        versioned_policy=versioned_policy,
    )

    summary = simulator.summarize(
        (
            RoutingSimulationCase(
                name="general-case",
                workload=LLMWorkload.GENERAL,
                expected_selected_model="primary",
            ),
        )
    )

    assert summary.policy_identifier == ("production-routing-policy@2.1.3")

    assert summary.total_cases == 1
    assert summary.passed is True
