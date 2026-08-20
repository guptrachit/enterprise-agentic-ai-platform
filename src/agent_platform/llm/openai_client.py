import time

from openai import AsyncOpenAI

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient, LLMMetadata, LLMResponse, LLMUsage
from agent_platform.llm.error_mapper import map_openai_error
from agent_platform.llm.errors import LLMError
from agent_platform.llm.retry import RetryPolicy
from agent_platform.llm.retry_executor import execute_with_retry


class OpenAIClient(LLMClient):
    """OpenAI implementation of the LLM client."""

    def __init__(self, settings: Settings) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-5-mini"
        self.retry_policy = RetryPolicy(
            max_attempts=settings.llm_max_retries + 1,
            initial_backoff_seconds=settings.llm_initial_backoff_seconds,
            max_backoff_seconds=settings.llm_max_backoff_seconds,
            attempt_timeout_seconds=settings.llm_timeout_seconds,
        )

    async def _request(self, prompt: str):
        """Execute one OpenAI API request."""

        try:
            return await self.client.responses.create(
                model=self.model,
                input=prompt,
            )
        except LLMError:
            raise
        except Exception as error:
            raise map_openai_error(error) from error

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate an LLM response with retry and normalized metadata."""

        start_time = time.perf_counter()

        response = await execute_with_retry(
            lambda: self._request(prompt),
            self.retry_policy,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        usage = response.usage

        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        return LLMResponse(
            text=response.output_text,
            usage=LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            metadata=LLMMetadata(
                provider="openai",
                model=self.model,
                latency_ms=latency_ms,
                request_id=response.id,
            ),
        )
