from collections.abc import Callable
from dataclasses import dataclass

from agent_platform.llm.base import LLMClient, LLMResponse
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_router import ModelRouter
from agent_platform.llm.routing_decision import RoutingDecision
from agent_platform.llm.telemetry import (
    create_routing_decision_event,
    log_routing_decision_event,
)


@dataclass(frozen=True)
class LLMExecutionResult:
    """LLM response together with its routing decision."""

    response: LLMResponse
    routing_decision: RoutingDecision
    executed_model: ModelDefinition
    fallback_used: bool


class LLMExecutionService:
    """Coordinates model routing and provider-neutral LLM execution."""

    def __init__(
        self,
        router: ModelRouter,
        client_factory: Callable[[ModelDefinition], LLMClient],
    ) -> None:
        self.router = router
        self.client_factory = client_factory

    async def execute(
        self,
        request: LLMExecutionRequest,
    ) -> LLMResponse:
        """Route and execute an LLM request."""

        result = await self.execute_with_decision(request)

        return result.response

    async def execute_with_decision(
        self,
        request: LLMExecutionRequest,
    ) -> LLMExecutionResult:
        """Execute an LLM request and expose its routing decision."""

        decision = self.router.route_decision(
            request.workload,
            constraints=request.constraints,
            required_capabilities=request.required_capabilities,
            preference=request.preference,
        )

        models = decision.ranked_candidates

        constraints = request.constraints
        preference = request.preference

        allowed_providers = (
            tuple(sorted(constraints.allowed_providers))
            if constraints is not None and constraints.allowed_providers is not None
            else None
        )

        max_cost_tier = (
            constraints.max_cost_tier.name.lower()
            if constraints is not None and constraints.max_cost_tier is not None
            else None
        )

        max_latency_tier = (
            constraints.max_latency_tier.name.lower()
            if constraints is not None and constraints.max_latency_tier is not None
            else None
        )

        prefer_lower_cost = (
            preference.prefer_lower_cost if preference is not None else False
        )

        prefer_lower_latency = (
            preference.prefer_lower_latency if preference is not None else False
        )

        preferred_providers = (
            preference.preferred_providers
            if preference is not None and preference.preferred_providers
            else None
        )

        preferred_cost_tier = (
            preference.preferred_cost_tier.name.lower()
            if preference is not None and preference.preferred_cost_tier is not None
            else None
        )

        preferred_latency_tier = (
            preference.preferred_latency_tier.name.lower()
            if preference is not None and preference.preferred_latency_tier is not None
            else None
        )

        routing_reason_codes = tuple(reason.code.value for reason in decision.reasons)

        routing_reasons = tuple(reason.message for reason in decision.reasons)

        ranked_candidates = tuple(
            candidate.name for candidate in decision.ranked_candidates
        )

        last_error: Exception | None = None
        last_attempted_model: ModelDefinition | None = None
        fallback_from: str | None = None
        fallback_reason: str | None = None

        for index, model in enumerate(models):
            last_attempted_model = model
            client = self.client_factory(model)

            try:
                response = await client.generate(
                    request.prompt,
                    correlation_id=request.correlation_id,
                    prompt_name=request.prompt_name,
                    prompt_version=request.prompt_version,
                    workload=request.workload.value,
                    logical_model=model.name,
                    fallback_used=index > 0,
                    fallback_from=fallback_from,
                    fallback_reason=fallback_reason,
                    allowed_providers=allowed_providers,
                    max_cost_tier=max_cost_tier,
                    max_latency_tier=max_latency_tier,
                    prefer_lower_cost=prefer_lower_cost,
                    prefer_lower_latency=prefer_lower_latency,
                    preferred_providers=preferred_providers,
                    preferred_cost_tier=preferred_cost_tier,
                    preferred_latency_tier=preferred_latency_tier,
                )

                log_routing_decision_event(
                    create_routing_decision_event(
                        workload=request.workload.value,
                        selected_model=decision.selected_model.name,
                        ranked_candidates=ranked_candidates,
                        rejected_models=decision.rejected_models,
                        routing_reason_codes=routing_reason_codes,
                        routing_reasons=routing_reasons,
                        executed_model=model.name,
                        fallback_used=index > 0,
                        success=True,
                        correlation_id=request.correlation_id,
                    )
                )

                return LLMExecutionResult(
                    response=response,
                    routing_decision=decision,
                    executed_model=model,
                    fallback_used=index > 0,
                )

            except Exception as error:
                last_error = error

                if not getattr(
                    error,
                    "retryable",
                    False,
                ):
                    log_routing_decision_event(
                        create_routing_decision_event(
                            workload=request.workload.value,
                            selected_model=decision.selected_model.name,
                            ranked_candidates=ranked_candidates,
                            rejected_models=decision.rejected_models,
                            routing_reason_codes=routing_reason_codes,
                            routing_reasons=routing_reasons,
                            executed_model=model.name,
                            fallback_used=index > 0,
                            success=False,
                            correlation_id=request.correlation_id,
                            error_type=type(error).__name__,
                        )
                    )

                    raise

                fallback_from = model.name
                fallback_reason = type(error).__name__

        if last_error is not None:
            log_routing_decision_event(
                create_routing_decision_event(
                    workload=request.workload.value,
                    selected_model=decision.selected_model.name,
                    ranked_candidates=ranked_candidates,
                    rejected_models=decision.rejected_models,
                    routing_reason_codes=routing_reason_codes,
                    routing_reasons=routing_reasons,
                    executed_model=(
                        last_attempted_model.name
                        if last_attempted_model is not None
                        else None
                    ),
                    fallback_used=len(models) > 1,
                    success=False,
                    correlation_id=request.correlation_id,
                    error_type=type(last_error).__name__,
                )
            )

            raise last_error

        raise RuntimeError("No routed model candidates were available.")
