from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    workload: str | None = None
    logical_model: str | None = None


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
        """Generate a provider-neutral LLM response."""
