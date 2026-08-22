from agent_platform.llm.workload import LLMWorkload


def test_workload_values_are_stable() -> None:
    assert LLMWorkload.GENERAL == "general"
    assert LLMWorkload.CLASSIFICATION == "classification"
    assert LLMWorkload.EXTRACTION == "extraction"
    assert LLMWorkload.REASONING == "reasoning"
