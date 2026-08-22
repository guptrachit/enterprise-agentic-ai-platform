from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.workload import LLMWorkload


def test_execution_request_defaults_to_general_workload() -> None:
    request = LLMExecutionRequest(
        prompt="Explain this concept.",
    )

    assert request.prompt == "Explain this concept."
    assert request.workload == LLMWorkload.GENERAL
    assert request.correlation_id is None
    assert request.prompt_name is None
    assert request.prompt_version is None


def test_execution_request_preserves_workload_and_metadata() -> None:
    request = LLMExecutionRequest(
        prompt="Classify this ticket.",
        workload=LLMWorkload.CLASSIFICATION,
        correlation_id="corr-123",
        prompt_name="ticket_classifier",
        prompt_version="2.0",
    )

    assert request.workload == LLMWorkload.CLASSIFICATION
    assert request.correlation_id == "corr-123"
    assert request.prompt_name == "ticket_classifier"
    assert request.prompt_version == "2.0"
