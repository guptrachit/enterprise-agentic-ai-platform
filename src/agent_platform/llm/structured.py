from dataclasses import dataclass

from pydantic import BaseModel

from agent_platform.llm.base import LLMMetadata, LLMUsage


@dataclass(frozen=True)
class StructuredLLMResponse[T: BaseModel]:
    """Normalized structured response returned by an LLM provider."""

    parsed: T
    usage: LLMUsage
    metadata: LLMMetadata
