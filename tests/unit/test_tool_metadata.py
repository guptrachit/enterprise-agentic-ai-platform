import pytest
from pydantic import ValidationError

from agent_platform.tools import (
    ToolDefinition,
    ToolInput,
    ToolMetadata,
    ToolOutput,
)


class SearchInput(ToolInput):
    query: str


class SearchOutput(ToolOutput):
    result: str


def test_tool_metadata_contains_stable_identity() -> None:
    metadata = ToolMetadata(
        name="enterprise_search",
        description="Search approved enterprise knowledge sources.",
        version="1.0.0",
    )

    assert metadata.name == "enterprise_search"
    assert metadata.description == "Search approved enterprise knowledge sources."
    assert metadata.version == "1.0.0"


def test_tool_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ToolMetadata(
            name="enterprise_search",
            description="Search approved enterprise knowledge sources.",
            version="1.0.0",
            unexpected_field="not-allowed",
        )


def test_tool_metadata_is_immutable() -> None:
    metadata = ToolMetadata(
        name="enterprise_search",
        description="Search approved enterprise knowledge sources.",
        version="1.0.0",
    )

    with pytest.raises(ValidationError):
        metadata.name = "changed"


def test_tool_definition_binds_metadata_and_contracts() -> None:
    metadata = ToolMetadata(
        name="enterprise_search",
        description="Search approved enterprise knowledge sources.",
        version="1.0.0",
    )

    definition = ToolDefinition[
        SearchInput,
        SearchOutput,
    ](
        metadata=metadata,
        input_type=SearchInput,
        output_type=SearchOutput,
    )

    assert definition.metadata == metadata
    assert definition.input_type is SearchInput
    assert definition.output_type is SearchOutput


def test_tool_definition_rejects_invalid_input_contract() -> None:
    class InvalidInput:
        pass

    metadata = ToolMetadata(
        name="enterprise_search",
        description="Search approved enterprise knowledge sources.",
        version="1.0.0",
    )

    with pytest.raises(TypeError, match="input_type must inherit from ToolInput"):
        ToolDefinition(
            metadata=metadata,
            input_type=InvalidInput,
            output_type=SearchOutput,
        )


def test_tool_definition_rejects_invalid_output_contract() -> None:
    class InvalidOutput:
        pass

    metadata = ToolMetadata(
        name="enterprise_search",
        description="Search approved enterprise knowledge sources.",
        version="1.0.0",
    )

    with pytest.raises(TypeError, match="output_type must inherit from ToolOutput"):
        ToolDefinition(
            metadata=metadata,
            input_type=SearchInput,
            output_type=InvalidOutput,
        )
