from agent_platform.llm.errors import (
    LLMModelConstraintViolationError,
    LLMModelDisabledError,
    LLMModelWorkloadNotSupportedError,
)
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_policy import ModelPolicy
from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.model_ranker import ModelRanker
from agent_platform.llm.model_registry import ModelRegistry
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.routing_decision import RoutingDecision
from agent_platform.llm.routing_reason import (
    RoutingReason,
    RoutingReasonCode,
)
from agent_platform.llm.workload import LLMWorkload


class ModelRouter:
    """Routes logical LLM workloads to valid model definitions."""

    def __init__(
        self,
        models: dict[str, ModelDefinition] | None = None,
        policy: ModelPolicy | None = None,
        registry: ModelRegistry | None = None,
        ranker: ModelRanker | None = None,
    ) -> None:
        if registry is not None:
            self.registry = registry
        else:
            self.registry = ModelRegistry(models=models)

        if policy is None:
            raise ValueError("Model policy is required.")

        self.policy = policy
        self.ranker = ranker or ModelRanker()

    def route(
        self,
        workload: LLMWorkload,
        constraints: RoutingConstraints | None = None,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
        preference: ModelPreference | None = None,
    ) -> ModelDefinition:
        """Resolve the highest-ranked valid model for a workload."""

        return self.route_candidates(
            workload,
            constraints=constraints,
            required_capabilities=required_capabilities,
            preference=preference,
        )[0]

    def route_candidates(
        self,
        workload: LLMWorkload,
        constraints: RoutingConstraints | None = None,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
        preference: ModelPreference | None = None,
    ) -> tuple[ModelDefinition, ...]:
        """Resolve and rank all valid model candidates for a workload."""

        eligible, _, _ = self._evaluate_candidates(
            workload,
            constraints=constraints,
            required_capabilities=required_capabilities,
        )

        if not eligible:
            self._raise_no_candidate_error(
                workload,
                constraints=constraints,
                required_capabilities=required_capabilities,
            )

        return self.ranker.rank(
            eligible,
            preference,
        )

    def route_decision(
        self,
        workload: LLMWorkload,
        constraints: RoutingConstraints | None = None,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
        preference: ModelPreference | None = None,
    ) -> RoutingDecision:
        """Resolve models and return an explainable routing decision."""

        eligible, rejected_models, reasons = self._evaluate_candidates(
            workload,
            constraints=constraints,
            required_capabilities=required_capabilities,
        )

        if not eligible:
            self._raise_no_candidate_error(
                workload,
                constraints=constraints,
                required_capabilities=required_capabilities,
            )

        ranked_candidates = self.ranker.rank(
            eligible,
            preference,
        )

        decision_reasons = list(reasons)

        if preference is not None:
            if preference.prefer_lower_cost:
                decision_reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.LOWER_COST_PREFERRED,
                        message=("Eligible models ranked with lower cost preferred"),
                    )
                )

            if preference.prefer_lower_latency:
                decision_reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.LOWER_LATENCY_PREFERRED,
                        message=("Eligible models ranked with lower latency preferred"),
                    )
                )

            if preference.preferred_providers:
                providers = ", ".join(preference.preferred_providers)

                decision_reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.PROVIDER_PREFERRED,
                        message=(
                            "Eligible models ranked using "
                            f"preferred providers: {providers}"
                        ),
                    )
                )

            if preference.preferred_cost_tier is not None:
                decision_reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.COST_TIER_PREFERRED,
                        message=(
                            "Eligible models ranked using "
                            "preferred cost tier: "
                            f"{preference.preferred_cost_tier.name.lower()}"
                        ),
                    )
                )

            if preference.preferred_latency_tier is not None:
                decision_reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.LATENCY_TIER_PREFERRED,
                        message=(
                            "Eligible models ranked using "
                            "preferred latency tier: "
                            f"{preference.preferred_latency_tier.name.lower()}"
                        ),
                    )
                )

        selected_model = ranked_candidates[0]

        decision_reasons.append(
            RoutingReason(
                code=RoutingReasonCode.SELECTED,
                message=(
                    f"Selected model '{selected_model.name}' "
                    "as the highest-ranked eligible candidate"
                ),
            )
        )

        return RoutingDecision(
            selected_model=selected_model,
            ranked_candidates=ranked_candidates,
            rejected_models=rejected_models,
            reasons=tuple(decision_reasons),
        )

    def _evaluate_candidates(
        self,
        workload: LLMWorkload,
        *,
        constraints: RoutingConstraints | None,
        required_capabilities: frozenset[ModelCapability],
    ) -> tuple[
        tuple[ModelDefinition, ...],
        tuple[str, ...],
        tuple[RoutingReason, ...],
    ]:
        """Evaluate policy candidates before preference ranking."""

        model_names = self.policy.models_for(workload)

        eligible: list[ModelDefinition] = []
        rejected_models: list[str] = []
        reasons: list[RoutingReason] = []

        for model_name in model_names:
            model = self.registry.get(model_name)

            if not model.enabled:
                rejected_models.append(model.name)

                reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.DISABLED,
                        message=(
                            f"Model '{model.name}' rejected because it is disabled"
                        ),
                    )
                )
                continue

            if not model.supports_workload(workload):
                rejected_models.append(model.name)

                reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.WORKLOAD_NOT_SUPPORTED,
                        message=(
                            f"Model '{model.name}' rejected "
                            "because it does not support "
                            f"workload '{workload.value}'"
                        ),
                    )
                )
                continue

            if constraints is not None and not constraints.allows(model):
                rejected_models.append(model.name)

                reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.CONSTRAINT_REJECTED,
                        message=(
                            f"Model '{model.name}' rejected by routing constraints"
                        ),
                    )
                )
                continue

            if not model.supports_capabilities(required_capabilities):
                rejected_models.append(model.name)

                missing_capabilities = sorted(
                    capability.value
                    for capability in (
                        required_capabilities - (model.capabilities or frozenset())
                    )
                )

                missing = ", ".join(missing_capabilities)

                reasons.append(
                    RoutingReason(
                        code=RoutingReasonCode.CAPABILITY_MISSING,
                        message=(
                            f"Model '{model.name}' rejected "
                            "because required capabilities "
                            f"are missing: {missing}"
                        ),
                    )
                )
                continue

            eligible.append(model)

        return (
            tuple(eligible),
            tuple(rejected_models),
            tuple(reasons),
        )

    def _raise_no_candidate_error(
        self,
        workload: LLMWorkload,
        *,
        constraints: RoutingConstraints | None,
        required_capabilities: frozenset[ModelCapability],
    ) -> None:
        """Raise the existing routing error for an empty candidate set."""

        model_names = self.policy.models_for(workload)
        primary_model = model_names[0]

        primary = self.registry.get(primary_model)

        if not primary.enabled:
            raise LLMModelDisabledError(primary_model)

        if not primary.supports_workload(workload):
            raise LLMModelWorkloadNotSupportedError(
                primary_model,
                workload.value,
            )

        if constraints is not None and not constraints.allows(primary):
            raise LLMModelConstraintViolationError(workload.value)

        if not primary.supports_capabilities(required_capabilities):
            raise LLMModelConstraintViolationError(workload.value)

        raise LLMModelWorkloadNotSupportedError(
            primary_model,
            workload.value,
        )
