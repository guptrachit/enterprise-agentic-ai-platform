from agent_platform.llm.api_models import (
    LLMGenerateRequest,
    LLMGenerateResponse,
)
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)


class GovernedLLMAPIService:
    """Translate API requests into governed LLM runtime execution."""

    def __init__(
        self,
        runtime: RefreshAwareLLMExecutionService,
    ) -> None:
        self.runtime = runtime

    async def generate(
        self,
        request: LLMGenerateRequest,
    ) -> LLMGenerateResponse:
        """Execute one governed LLM generation request."""

        execution_result = await self.runtime.execute_with_decision(
            LLMExecutionRequest(
                prompt=request.prompt,
                workload=request.workload,
                correlation_id=request.correlation_id,
                prompt_name=request.prompt_name,
                prompt_version=request.prompt_version,
            )
        )

        return LLMGenerateResponse(
            content=execution_result.response.content,
            policy_identifier=self.runtime.policy_identifier,
            model=execution_result.executed_model.name,
            provider=execution_result.executed_model.provider,
        )
