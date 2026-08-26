from enum import StrEnum


class SecurityScope(StrEnum):
    """Centralized authorization scopes for platform APIs."""

    LLM_GENERATE = "llm:generate"
    LLM_HEALTH = "llm:health"

    AGENT_EXECUTE = "agent:execute"
    TOOL_EXECUTE = "tool:execute"

    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
