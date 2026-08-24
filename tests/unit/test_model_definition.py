from agent_platform.llm.model_capability import ModelCapability
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


def test_model_definition_supports_required_capabilities() -> None:
    model = ModelDefinition(
        name="capable_model",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.TOOL_CALLING,
            }
        ),
    )

    assert model.supports_capabilities(
        frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
            }
        )
    )


def test_model_definition_rejects_missing_capability() -> None:
    model = ModelDefinition(
        name="text_model",
        provider="openai",
        provider_model="gpt-5-mini",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        capabilities=frozenset(
            {
                ModelCapability.STRUCTURED_OUTPUT,
            }
        ),
    )

    assert not model.supports_capabilities(
        frozenset(
            {
                ModelCapability.VISION,
            }
        )
    )


def test_legacy_structured_output_flag_creates_capability() -> None:
    model = ModelDefinition(
        name="legacy_model",
        provider="openai",
        provider_model="legacy-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        supports_structured_output=True,
    )

    assert model.supports_structured_output is True

    assert ModelCapability.STRUCTURED_OUTPUT in model.capabilities


def test_explicit_capabilities_override_legacy_structured_output_flag() -> None:
    model = ModelDefinition(
        name="tool_only_model",
        provider="openai",
        provider_model="tool-model",
        workloads=frozenset(
            {
                LLMWorkload.GENERAL,
            }
        ),
        supports_structured_output=True,
        capabilities=frozenset(
            {
                ModelCapability.TOOL_CALLING,
            }
        ),
    )

    assert model.supports_structured_output is False

    assert model.capabilities == frozenset(
        {
            ModelCapability.TOOL_CALLING,
        }
    )
