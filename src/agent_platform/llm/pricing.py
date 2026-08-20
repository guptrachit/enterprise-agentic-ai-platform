from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float


PRICING: dict[str, ModelPricing] = {
    "gpt-5-mini": ModelPricing(
        input_cost_per_million_tokens=0.25,
        output_cost_per_million_tokens=2.00,
    ),
}
