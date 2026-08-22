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
        """Route and execute an LLM request."""

        model = self.router.route(request.workload)

        client = self.client_factory(model)

        return await client.generate(
            request.prompt,
            correlation_id=request.correlation_id,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
        )
