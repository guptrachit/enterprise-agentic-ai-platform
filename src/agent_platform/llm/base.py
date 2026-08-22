from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agent_platform.llm.prompt import PromptTemplate
    from agent_platform.llm.structured import StructuredLLMResponse


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMMetadata:
    provider: str
    model: str
    latency_ms: float
    request_id: str | None
    correlation_id: str
    retry_count: int
    estimated_cost_usd: float
    prompt_name: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage
    metadata: LLMMetadata


class LLMClient(ABC):
    """Abstract interface for interacting with an LLM provider."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        correlation_id: str | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError

    async def generate_from_template(
        self,
        prompt_template: "PromptTemplate",
        variables: dict[str, object],
        *,
        correlation_id: str | None = None,
    ) -> LLMResponse:
        """Render and execute a managed prompt template."""

        rendered_prompt = prompt_template.render(**variables)

        return await self.generate(
            rendered_prompt,
            correlation_id=correlation_id,
            prompt_name=prompt_template.name,
            prompt_version=prompt_template.version,
        )

    @abstractmethod
    async def generate_structured[T: BaseModel](
        self,
        prompt: str,
        response_model: type[T],
        *,
        correlation_id: str | None = None,
    ) -> "StructuredLLMResponse[T]":
        """Generate and validate a structured LLM response."""
        raise NotImplementedError
