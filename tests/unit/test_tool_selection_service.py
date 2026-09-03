from typing import Any

import pytest

from agent_platform.tools import (
    ToolDefinition,
    ToolInput,
    ToolMetadata,
    ToolOutput,
    ToolRegistry,
    ToolSelection,
    ToolSelectionError,
    ToolSelectionRequest,
    ToolSelectionService,
)


class SearchInput(ToolInput):
    query: str


class SearchOutput(ToolOutput):
    result: str


def create_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            metadata=ToolMetadata(
                name="search",
                description="Search enterprise information.",
                version="1.0.0",
            ),
            input_type=SearchInput,
            output_type=SearchOutput,
        )
    )

    return registry


class SelectingModel:
    async def select_tool(
        self,
        *,
        user_message: str,
        tools: tuple[dict[str, Any], ...],
    ) -> ToolSelection | None:
        assert user_message == "Find customer 123"
        assert tools[0]["name"] == "search"

        return ToolSelection(
            tool_name="search",
            tool_version="1.0.0",
            arguments={
                "query": "customer 123",
            },
        )


class NoSelectionModel:
    async def select_tool(
        self,
        *,
        user_message: str,
        tools: tuple[dict[str, Any], ...],
    ) -> ToolSelection | None:
        return None


class UnknownToolModel:
    async def select_tool(
        self,
        *,
        user_message: str,
        tools: tuple[dict[str, Any], ...],
    ) -> ToolSelection | None:
        return ToolSelection(
            tool_name="delete_everything",
            tool_version="1.0.0",
            arguments={},
        )


@pytest.mark.asyncio
async def test_selection_service_returns_registered_tool() -> None:
    service = ToolSelectionService(
        registry=create_registry(),
        model=SelectingModel(),
    )

    selection = await service.select(
        request=ToolSelectionRequest(
            user_message="Find customer 123",
        )
    )

    assert selection is not None
    assert selection.tool_name == "search"
    assert selection.tool_version == "1.0.0"
    assert selection.arguments == {
        "query": "customer 123",
    }


@pytest.mark.asyncio
async def test_selection_service_allows_no_tool_selection() -> None:
    service = ToolSelectionService(
        registry=create_registry(),
        model=NoSelectionModel(),
    )

    selection = await service.select(
        request=ToolSelectionRequest(
            user_message="Hello",
        )
    )

    assert selection is None


@pytest.mark.asyncio
async def test_selection_service_rejects_unknown_tool() -> None:
    service = ToolSelectionService(
        registry=create_registry(),
        model=UnknownToolModel(),
    )

    with pytest.raises(
        ToolSelectionError,
        match="unknown tool",
    ):
        await service.select(
            request=ToolSelectionRequest(
                user_message="Delete everything",
            )
        )
