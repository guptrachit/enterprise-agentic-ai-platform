from pydantic import BaseModel, Field

from agent_platform.llm.workload import LLMWorkload


class ModelPolicyConfig(BaseModel):
    """Configuration representation of workload-to-model routing policy."""

    assignments: dict[
        LLMWorkload,
        str | tuple[str, ...],
    ] = Field(default_factory=dict)
