from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.workload import LLMWorkload


def test_model_definition_supports_declared_workload() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
                LLMWorkload.CLASSIFICATION,
            }
        ),
    )

    assert model.supports_workload(LLMWorkload.CLASSIFICATION)


def test_model_definition_rejects_undeclared_workload() -> None:
    model = ModelDefinition(
        name="fast_general",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )

    assert not model.supports_workload(LLMWorkload.REASONING)


def test_model_definition_can_disable_model() -> None:
    model = ModelDefinition(
        name="disabled_model",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        enabled=False,
    )

    assert model.enabled is False
