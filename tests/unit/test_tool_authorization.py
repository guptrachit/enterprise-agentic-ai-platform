import pytest

from agent_platform.tools import (
    ToolAuthorizationContext,
    ToolAuthorizationDeniedError,
    ToolAuthorizationPolicy,
    ToolAuthorizationRule,
    ToolDefinition,
    ToolInput,
    ToolMetadata,
    ToolOutput,
)


class SearchInput(ToolInput):
    query: str


class SearchOutput(ToolOutput):
    result: str


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


def test_authorization_allows_subject_with_required_scope() -> None:
    policy = ToolAuthorizationPolicy(
        rules=(
            ToolAuthorizationRule(
                tool_name="enterprise_search",
                tool_version="1.0.0",
                required_scopes=frozenset(
                    {
                        "tool:enterprise_search:execute",
                    }
                ),
            ),
        )
    )

    policy.authorize(
        definition=create_definition(),
        context=ToolAuthorizationContext(
            subject_id="user-123",
            granted_scopes=frozenset(
                {
                    "tool:enterprise_search:execute",
                }
            ),
        ),
    )


def test_authorization_denies_missing_required_scope() -> None:
    policy = ToolAuthorizationPolicy(
        rules=(
            ToolAuthorizationRule(
                tool_name="enterprise_search",
                tool_version="1.0.0",
                required_scopes=frozenset(
                    {
                        "tool:enterprise_search:execute",
                    }
                ),
            ),
        )
    )

    with pytest.raises(
        ToolAuthorizationDeniedError,
        match="missing scopes",
    ):
        policy.authorize(
            definition=create_definition(),
            context=ToolAuthorizationContext(
                subject_id="user-123",
                granted_scopes=frozenset(),
            ),
        )


def test_authorization_denies_tool_without_policy_rule() -> None:
    policy = ToolAuthorizationPolicy()

    with pytest.raises(
        ToolAuthorizationDeniedError,
        match="no authorization rule configured",
    ):
        policy.authorize(
            definition=create_definition(),
            context=ToolAuthorizationContext(
                subject_id="user-123",
                granted_scopes=frozenset(
                    {
                        "tool:enterprise_search:execute",
                    }
                ),
            ),
        )


def test_authorization_requires_all_configured_scopes() -> None:
    policy = ToolAuthorizationPolicy(
        rules=(
            ToolAuthorizationRule(
                tool_name="enterprise_search",
                tool_version="1.0.0",
                required_scopes=frozenset(
                    {
                        "tool:enterprise_search:execute",
                        "enterprise:data:read",
                    }
                ),
            ),
        )
    )

    with pytest.raises(
        ToolAuthorizationDeniedError,
        match="enterprise:data:read",
    ):
        policy.authorize(
            definition=create_definition(),
            context=ToolAuthorizationContext(
                subject_id="user-123",
                granted_scopes=frozenset(
                    {
                        "tool:enterprise_search:execute",
                    }
                ),
            ),
        )
