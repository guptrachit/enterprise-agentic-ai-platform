import pytest

from agent_platform.tools import (
    ToolDefinition,
    ToolInput,
    ToolInputValidationError,
    ToolMetadata,
    ToolOutput,
    ToolOutputValidationError,
    ToolValidator,
)


class SearchInput(ToolInput):
    query: str
    limit: int


class SearchOutput(ToolOutput):
    results: list[str]
    count: int


def create_definition() -> ToolDefinition[SearchInput, SearchOutput]:
    return ToolDefinition(
        metadata=ToolMetadata(
            name="enterprise_search",
            description="Search approved enterprise knowledge sources.",
            version="1.0.0",
        ),
        input_type=SearchInput,
        output_type=SearchOutput,
    )


def test_validator_accepts_valid_tool_input() -> None:
    validator = ToolValidator()
    definition = create_definition()

    result = validator.validate_input(
        definition=definition,
        payload={
            "query": "customer policy",
            "limit": 5,
        },
    )

    assert isinstance(result, SearchInput)
    assert result.query == "customer policy"
    assert result.limit == 5


def test_validator_rejects_missing_required_input() -> None:
    validator = ToolValidator()
    definition = create_definition()

    with pytest.raises(
        ToolInputValidationError,
        match="enterprise_search:1.0.0",
    ):
        validator.validate_input(
            definition=definition,
            payload={
                "query": "customer policy",
            },
        )


def test_validator_rejects_unknown_input_fields() -> None:
    validator = ToolValidator()
    definition = create_definition()

    with pytest.raises(
        ToolInputValidationError,
        match="enterprise_search:1.0.0",
    ):
        validator.validate_input(
            definition=definition,
            payload={
                "query": "customer policy",
                "limit": 5,
                "delete_everything": True,
            },
        )


def test_validator_rejects_invalid_input_type() -> None:
    validator = ToolValidator()
    definition = create_definition()

    with pytest.raises(
        ToolInputValidationError,
        match="enterprise_search:1.0.0",
    ):
        validator.validate_input(
            definition=definition,
            payload={
                "query": "customer policy",
                "limit": "not-an-integer",
            },
        )


def test_validator_accepts_valid_tool_output() -> None:
    validator = ToolValidator()
    definition = create_definition()

    result = validator.validate_output(
        definition=definition,
        payload={
            "results": ["policy-a", "policy-b"],
            "count": 2,
        },
    )

    assert isinstance(result, SearchOutput)
    assert result.results == ["policy-a", "policy-b"]
    assert result.count == 2


def test_validator_rejects_invalid_tool_output() -> None:
    validator = ToolValidator()
    definition = create_definition()

    with pytest.raises(
        ToolOutputValidationError,
        match="enterprise_search:1.0.0",
    ):
        validator.validate_output(
            definition=definition,
            payload={
                "results": ["policy-a"],
                "count": "invalid-count",
            },
        )


def test_validator_rejects_unknown_output_fields() -> None:
    validator = ToolValidator()
    definition = create_definition()

    with pytest.raises(
        ToolOutputValidationError,
        match="enterprise_search:1.0.0",
    ):
        validator.validate_output(
            definition=definition,
            payload={
                "results": ["policy-a"],
                "count": 1,
                "unexpected": "not-allowed",
            },
        )
