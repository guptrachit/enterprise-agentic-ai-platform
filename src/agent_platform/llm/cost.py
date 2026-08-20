from agent_platform.llm.base import LLMUsage
from agent_platform.llm.pricing import PRICING


def calculate_cost(model: str, usage: LLMUsage) -> float:
    pricing = PRICING[model]

    input_cost = (
        usage.input_tokens / 1_000_000
    ) * pricing.input_cost_per_million_tokens

    output_cost = (
        usage.output_tokens / 1_000_000
    ) * pricing.output_cost_per_million_tokens

    return input_cost + output_cost
