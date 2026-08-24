from collections.abc import Callable

from agent_platform.llm.base import LLMClient, LLMResponse
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_router import ModelRouter


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
        """Route and execute an LLM request with eligible model fallback."""

        models = self.router.route_candidates(
            request.workload,
            constraints=request.constraints,
            required_capabilities=request.required_capabilities,
        )

        constraints = request.constraints

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

        last_error: Exception | None = None
        fallback_from: str | None = None
        fallback_reason: str | None = None

        for index, model in enumerate(models):
            client = self.client_factory(model)

            try:
                return await client.generate(
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
                )
            except Exception as error:
                last_error = error

                if not getattr(
                    error,
                    "retryable",
                    False,
                ):
                    raise

                fallback_from = model.name
                fallback_reason = type(error).__name__

        if last_error is not None:
            raise last_error

        raise RuntimeError("No routed model candidates were available.")
