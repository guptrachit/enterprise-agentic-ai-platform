from agent_platform.tools import (
    ToolDefinition,
    ToolInput,
    ToolMetadata,
    ToolOutput,
    ToolSchemaBuilder,
)


class SearchInput(ToolInput):
    query: str


class SearchOutput(ToolOutput):
    result: str


def create_definition() -> ToolDefinition[SearchInput, SearchOutput]:
    return ToolDefinition(
        metadata=ToolMetadata(
            name="search",
            description="Search enterprise information.",
            version="1.0.0",
        ),
        input_type=SearchInput,
        output_type=SearchOutput,
    )


def test_schema_builder_exposes_tool_identity() -> None:
    schema = ToolSchemaBuilder().build(
        definition=create_definition(),
    )

    assert schema["name"] == "search"
    assert schema["version"] == "1.0.0"
    assert schema["description"] == "Search enterprise information."


def test_schema_builder_uses_input_contract_schema() -> None:
    schema = ToolSchemaBuilder().build(
        definition=create_definition(),
    )

    input_schema = schema["input_schema"]

    assert "query" in input_schema["properties"]
    assert "query" in input_schema["required"]


def test_schema_builder_builds_multiple_definitions() -> None:
    definition = create_definition()

    schemas = ToolSchemaBuilder().build_many(
        definitions=(definition,),
    )

    assert len(schemas) == 1
    assert schemas[0]["name"] == "search"
