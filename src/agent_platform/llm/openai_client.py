import time
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient, LLMMetadata, LLMResponse, LLMUsage
from agent_platform.llm.cost import calculate_cost
from agent_platform.llm.error_mapper import map_openai_error
from agent_platform.llm.errors import (
    LLMError,
    LLMRefusalError,
    LLMStructuredParseError,
    LLMStructuredValidationError,
)
from agent_platform.llm.prompt import PromptTemplate
from agent_platform.llm.retry import RetryPolicy
from agent_platform.llm.retry_executor import execute_with_retry
from agent_platform.llm.structured import StructuredLLMResponse
from agent_platform.llm.telemetry import (
    create_execution_event,
    log_execution_event,
)


def _extract_refusal(response) -> str | None:
    """Return provider refusal text when present."""

    for output in getattr(response, "output", []):
        if getattr(output, "type", None) != "message":
            continue

        for item in getattr(output, "content", []):
            if getattr(item, "type", None) == "refusal":
                return getattr(item, "refusal", None)

    return None


class OpenAIClient(LLMClient):
    """OpenAI implementation of the LLM client."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.llm_model
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
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        workload: str | None = None,
        logical_model: str | None = None,
        fallback_used: bool = False,
        fallback_from: str | None = None,
        fallback_reason: str | None = None,
        allowed_providers: tuple[str, ...] | None = None,
        max_cost_tier: str | None = None,
        max_latency_tier: str | None = None,
        prefer_lower_cost: bool = False,
        prefer_lower_latency: bool = False,
        preferred_providers: tuple[str, ...] | None = None,
        preferred_cost_tier: str | None = None,
        preferred_latency_tier: str | None = None,
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
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    workload=workload,
                    logical_model=logical_model,
                    fallback_used=fallback_used,
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
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    workload=workload,
                    logical_model=logical_model,
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
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    workload=workload,
                    logical_model=logical_model,
                    error_type=type(error).__name__,
                    fallback_used=fallback_used,
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
            )

            raise

    async def generate_from_template(
        self,
        template: PromptTemplate,
        variables: dict[str, object],
        *,
        correlation_id: str | None = None,
        workload: str | None = None,
        logical_model: str | None = None,
        fallback_used: bool = False,
        fallback_from: str | None = None,
        fallback_reason: str | None = None,
        allowed_providers: tuple[str, ...] | None = None,
        max_cost_tier: str | None = None,
        max_latency_tier: str | None = None,
        prefer_lower_cost: bool = False,
        prefer_lower_latency: bool = False,
        preferred_providers: tuple[str, ...] | None = None,
        preferred_cost_tier: str | None = None,
        preferred_latency_tier: str | None = None,
    ) -> LLMResponse:
        """Render a versioned prompt template and generate a response."""

        prompt = template.render(**variables)

        return await self.generate(
            prompt,
            correlation_id=correlation_id,
            prompt_name=template.name,
            prompt_version=template.version,
            workload=workload,
            logical_model=logical_model,
            fallback_used=fallback_used,
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

    async def generate_structured_from_template[T: BaseModel](
        self,
        template: PromptTemplate,
        response_model: type[T],
        variables: dict[str, object],
        *,
        correlation_id: str | None = None,
    ) -> StructuredLLMResponse[T]:
        """Render a versioned prompt template and generate structured output."""

        prompt = template.render(**variables)

        return await self.generate_structured(
            prompt,
            response_model,
            correlation_id=correlation_id,
            prompt_name=template.name,
            prompt_version=template.version,
        )

    async def generate_structured[T: BaseModel](
        self,
        prompt: str,
        response_model: type[T],
        *,
        correlation_id: str | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
    ) -> StructuredLLMResponse[T]:
        """Generate a validated structured response."""

        correlation_id = correlation_id or str(uuid4())
        start_time = time.perf_counter()

        async def request():
            try:
                return await self.client.responses.parse(
                    model=self.model,
                    input=prompt,
                    text_format=response_model,
                )
            except LLMError:
                raise
            except ValidationError as error:
                raise LLMStructuredValidationError() from error
            except Exception as error:
                raise map_openai_error(error) from error

        execution = await execute_with_retry(
            request,
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

        refusal = _extract_refusal(response)

        if refusal is not None:
            raise LLMRefusalError(refusal)

        parsed = response.output_parsed

        if parsed is None:
            raise LLMStructuredParseError()

        metadata = LLMMetadata(
            provider="openai",
            model=self.model,
            latency_ms=latency_ms,
            request_id=response.id,
            correlation_id=correlation_id,
            retry_count=retry_count,
            estimated_cost_usd=estimated_cost_usd,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )

        return StructuredLLMResponse(
            parsed=parsed,
            usage=llm_usage,
            metadata=metadata,
        )
