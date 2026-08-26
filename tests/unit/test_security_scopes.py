from agent_platform.security.scopes import (
    SecurityScope,
)


def test_llm_generate_scope_value() -> None:
    assert SecurityScope.LLM_GENERATE.value == "llm:generate"


def test_llm_health_scope_value() -> None:
    assert SecurityScope.LLM_HEALTH.value == "llm:health"


def test_agent_execute_scope_value() -> None:
    assert SecurityScope.AGENT_EXECUTE.value == "agent:execute"


def test_tool_execute_scope_value() -> None:
    assert SecurityScope.TOOL_EXECUTE.value == "tool:execute"


def test_admin_scope_values() -> None:
    assert SecurityScope.ADMIN_READ.value == "admin:read"
    assert SecurityScope.ADMIN_WRITE.value == "admin:write"
