import pytest

from agent_platform.tools import (
    DuplicateToolError,
    ToolDefinition,
    ToolInput,
    ToolMetadata,
    ToolNotFoundError,
    ToolOutput,
    ToolRegistry,
)


class SearchInput(ToolInput):
    query: str


class SearchOutput(ToolOutput):
    result: str


def create_definition(
    *,
    name: str = "enterprise_search",
    version: str = "1.0.0",
) -> ToolDefinition[SearchInput, SearchOutput]:
    return ToolDefinition(
        metadata=ToolMetadata(
            name=name,
            description="Search approved enterprise knowledge sources.",
            version=version,
        ),
        input_type=SearchInput,
        output_type=SearchOutput,
    )


def test_registry_registers_tool_definition() -> None:
    registry = ToolRegistry()
    definition = create_definition()

    registry.register(definition)

    assert registry.contains(
        name="enterprise_search",
        version="1.0.0",
    )


def test_registry_returns_registered_tool_definition() -> None:
    registry = ToolRegistry()
    definition = create_definition()

    registry.register(definition)

    result = registry.get(
        name="enterprise_search",
        version="1.0.0",
    )

    assert result is definition


def test_registry_rejects_duplicate_name_and_version() -> None:
    registry = ToolRegistry()

    registry.register(create_definition())

    with pytest.raises(
        DuplicateToolError,
        match="enterprise_search:1.0.0",
    ):
        registry.register(create_definition())


def test_registry_allows_multiple_versions_of_same_tool() -> None:
    registry = ToolRegistry()

    registry.register(
        create_definition(
            version="1.0.0",
        )
    )

    registry.register(
        create_definition(
            version="2.0.0",
        )
    )

    assert registry.contains(
        name="enterprise_search",
        version="1.0.0",
    )

    assert registry.contains(
        name="enterprise_search",
        version="2.0.0",
    )


def test_registry_raises_when_tool_is_not_registered() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError,
        match="enterprise_search:1.0.0",
    ):
        registry.get(
            name="enterprise_search",
            version="1.0.0",
        )


def test_registry_lists_registered_definitions() -> None:
    registry = ToolRegistry()

    version_one = create_definition(
        version="1.0.0",
    )

    version_two = create_definition(
        version="2.0.0",
    )

    registry.register(version_one)
    registry.register(version_two)

    definitions = registry.list_definitions()

    assert definitions == (
        version_one,
        version_two,
    )
