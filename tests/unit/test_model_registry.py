import pytest

from agent_platform.llm.errors import LLMModelNotFoundError
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_registry import ModelRegistry
from agent_platform.llm.workload import LLMWorkload


def create_model(
    name: str = "fast_general",
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
    )


def test_model_registry_registers_and_returns_model() -> None:
    registry = ModelRegistry()

    model = create_model()

    registry.register(model)

    assert registry.get("fast_general") is model


def test_model_registry_can_be_initialized_with_models() -> None:
    model = create_model()

    registry = ModelRegistry(
        models={
            "fast_general": model,
        }
    )

    assert registry.get("fast_general") is model


def test_model_registry_raises_when_model_not_found() -> None:
    registry = ModelRegistry()

    with pytest.raises(
        LLMModelNotFoundError,
        match="missing_model",
    ):
        registry.get("missing_model")


def test_model_registry_returns_all_models() -> None:
    first = create_model("first")
    second = create_model("second")

    registry = ModelRegistry(
        models={
            "first": first,
            "second": second,
        }
    )

    assert registry.all() == (
        first,
        second,
    )


def test_model_registry_returns_copy_as_dict() -> None:
    model = create_model()

    registry = ModelRegistry(
        models={
            "fast_general": model,
        }
    )

    models = registry.as_dict()

    assert models == {
        "fast_general": model,
    }

    models.clear()

    assert registry.get("fast_general") is model
