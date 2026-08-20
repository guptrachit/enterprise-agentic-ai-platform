from agent_platform.llm.base import LLMUsage
from agent_platform.llm.cost import calculate_cost


def test_calculate_cost() -> None:
    usage = LLMUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    cost = calculate_cost("gpt-5-mini", usage)

    assert cost == 2.25
