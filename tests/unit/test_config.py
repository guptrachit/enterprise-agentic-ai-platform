from agent_platform.config import get_settings


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.app_env == "development"
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key
