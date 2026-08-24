from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.model_tier import ModelCostTier
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.workload import LLMWorkload


def test_execution_request_defaults() -> None:
    request = LLMExecutionRequest(
        prompt="Hello.",
    )

    assert request.prompt == "Hello."
    assert request.workload is LLMWorkload.GENERAL
    assert request.correlation_id is None
    assert request.prompt_name is None
    assert request.prompt_version is None
    assert request.constraints is None
    assert request.required_capabilities == frozenset()
    assert request.preference is None


def test_execution_request_preserves_metadata() -> None:
    request = LLMExecutionRequest(
        prompt="Classify this ticket.",
        workload=LLMWorkload.CLASSIFICATION,
        correlation_id="corr-123",
        prompt_name="ticket_classifier",
        prompt_version="2.0",
    )

    assert request.prompt == "Classify this ticket."
    assert request.workload is LLMWorkload.CLASSIFICATION
    assert request.correlation_id == "corr-123"
    assert request.prompt_name == "ticket_classifier"
    assert request.prompt_version == "2.0"


def test_execution_request_preserves_routing_constraints() -> None:
    constraints = RoutingConstraints(
        allowed_providers=frozenset(
            {
                "openai",
            }
        ),
        max_cost_tier=ModelCostTier.LOW,
    )

    request = LLMExecutionRequest(
        prompt="Classify this ticket.",
        workload=LLMWorkload.CLASSIFICATION,
        constraints=constraints,
    )

    assert request.constraints is constraints


def test_execution_request_preserves_required_capabilities() -> None:
    required = frozenset(
        {
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.TOOL_CALLING,
        }
    )

    request = LLMExecutionRequest(
        prompt="Process this request.",
        workload=LLMWorkload.GENERAL,
        required_capabilities=required,
    )

    assert request.required_capabilities == required


def test_execution_request_preserves_model_preference() -> None:
    preference = ModelPreference(
        prefer_lower_cost=True,
        preferred_providers=("openai",),
    )

    request = LLMExecutionRequest(
        prompt="Process this request.",
        workload=LLMWorkload.GENERAL,
        preference=preference,
    )

    assert request.preference is preference
