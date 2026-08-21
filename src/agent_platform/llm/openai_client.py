import time
from uuid import uuid4

from openai import AsyncOpenAI

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient, LLMMetadata, LLMResponse, LLMUsage
from agent_platform.llm.cost import calculate_cost
from agent_platform.llm.error_mapper import map_openai_error
from agent_platform.llm.errors import LLMError
from agent_platform.llm.retry import RetryPolicy
from agent_platform.llm.retry_executor import execute_with_retry
from agent_platform.llm.telemetry import (
    create_execution_event,
    log_execution_event,
)


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

    async def generate(
        self,
        prompt: str,
        *,
        correlation_id: str | None = None,
    ) -> LLMResponse:
        """Generate an LLM response with retry and normalized metadata."""

        correlation_id = correlation_id or str(uuid4())
        start_time = time.perf_counter()

        try:
            execution = await execute_with_retry(
                lambda: self._request(prompt),
                self.retry_policy,
            )

            response = execution.result
            retry_count = execution.retry_count

            latency_ms = (time.perf_counter() - start_time) * 1000

            usage = response.usage

            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0

            llm_usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

            estimated_cost_usd = calculate_cost(
                self.model,
                llm_usage,
            )

            log_execution_event(
                create_execution_event(
                    provider="openai",
                    model=self.model,
                    request_id=response.id,
                    correlation_id=correlation_id,
                    success=True,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    input_tokens=llm_usage.input_tokens,
                    output_tokens=llm_usage.output_tokens,
                    total_tokens=llm_usage.total_tokens,
                    estimated_cost_usd=estimated_cost_usd,
                )
            )

            return LLMResponse(
                text=response.output_text,
                usage=llm_usage,
                metadata=LLMMetadata(
                    provider="openai",
                    model=self.model,
                    latency_ms=latency_ms,
                    request_id=response.id,
                    correlation_id=correlation_id,
                    retry_count=retry_count,
                    estimated_cost_usd=estimated_cost_usd,
                ),
            )

        except LLMError as error:
            latency_ms = (time.perf_counter() - start_time) * 1000

            log_execution_event(
                create_execution_event(
                    provider="openai",
                    model=self.model,
                    request_id=None,
                    correlation_id=correlation_id,
                    success=False,
                    latency_ms=latency_ms,
                    retry_count=None,
                    error_type=type(error).__name__,
                )
            )

            raise
