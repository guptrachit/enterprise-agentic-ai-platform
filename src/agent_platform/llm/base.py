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
    retry_count: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage
    metadata: LLMMetadata


class LLMClient(ABC):
    """Abstract interface for interacting with an LLM provider."""

    @abstractmethod
    async def generate(self, prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError
