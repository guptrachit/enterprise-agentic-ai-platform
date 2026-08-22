from pydantic import BaseModel

from agent_platform.llm.base import LLMClient, LLMResponse
from agent_platform.llm.prompt_environment import PromptEnvironment
from agent_platform.llm.prompt_registry import PromptRegistry
from agent_platform.llm.structured import StructuredLLMResponse


class PromptExecutionService:
    """Coordinates managed prompt lookup and LLM execution."""

    def __init__(
        self,
        registry: PromptRegistry,
        client: LLMClient,
    ) -> None:
        self.registry = registry
        self.client = client

    async def generate_active(
        self,
        name: str,
        variables: dict[str, object],
        *,
        environment: PromptEnvironment = PromptEnvironment.PRODUCTION,
        correlation_id: str | None = None,
    ) -> LLMResponse:
        """Execute the active managed prompt for an environment."""

        prompt = self.registry.get_active(
            name,
            environment=environment,
        )

        return await self.client.generate_from_template(
            prompt,
            variables,
            correlation_id=correlation_id,
        )

    async def generate_structured_active[T: BaseModel](
        self,
        name: str,
        response_model: type[T],
        variables: dict[str, object],
        *,
        environment: PromptEnvironment = PromptEnvironment.PRODUCTION,
        correlation_id: str | None = None,
    ) -> StructuredLLMResponse[T]:
        """Execute the active structured prompt for an environment."""

        prompt = self.registry.get_active(
            name,
            environment=environment,
        )

        return await self.client.generate_structured_from_template(
            prompt,
            response_model,
            variables,
            correlation_id=correlation_id,
        )
