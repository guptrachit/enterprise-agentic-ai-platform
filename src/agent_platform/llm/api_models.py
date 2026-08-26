from pydantic import BaseModel, Field

from agent_platform.llm.workload import LLMWorkload


class LLMGenerateRequest(BaseModel):
    """API request for governed LLM execution."""

    prompt: str = Field(
        min_length=1,
    )
    workload: LLMWorkload = LLMWorkload.GENERAL
    correlation_id: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None


class LLMGenerateResponse(BaseModel):
    """API response from governed LLM execution."""

    content: str
    policy_identifier: str | None
    model: str | None
    provider: str | None
