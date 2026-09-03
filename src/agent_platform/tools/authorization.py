from dataclasses import dataclass

from agent_platform.tools.definition import ToolDefinition


@dataclass(frozen=True)
class ToolAuthorizationContext:
    """Authorization information carried into governed tool execution."""

    subject_id: str
    granted_scopes: frozenset[str]


@dataclass(frozen=True)
class ToolAuthorizationRule:
    """Authorization requirements for a specific tool identity."""

    tool_name: str
    tool_version: str
    required_scopes: frozenset[str]


class ToolAuthorizationError(Exception):
    """Base error raised by tool authorization."""


class ToolAuthorizationDeniedError(ToolAuthorizationError):
    """Raised when the caller is not permitted to execute a tool."""


class ToolAuthorizationPolicy:
    """Evaluates whether a caller may execute a governed tool."""

    def __init__(
        self,
        *,
        rules: tuple[ToolAuthorizationRule, ...] = (),
    ) -> None:
        self._rules = {(rule.tool_name, rule.tool_version): rule for rule in rules}

    def authorize(
        self,
        *,
        definition: ToolDefinition,
        context: ToolAuthorizationContext,
    ) -> None:
        key = (
            definition.metadata.name,
            definition.metadata.version,
        )

        rule = self._rules.get(key)

        if rule is None:
            raise ToolAuthorizationDeniedError(
                f"no authorization rule configured for tool "
                f"{definition.metadata.name}:{definition.metadata.version}"
            )

        missing_scopes = rule.required_scopes - context.granted_scopes

        if missing_scopes:
            missing = ", ".join(sorted(missing_scopes))

            raise ToolAuthorizationDeniedError(
                f"subject {context.subject_id} is not authorized for tool "
                f"{definition.metadata.name}:{definition.metadata.version}; "
                f"missing scopes: {missing}"
            )
