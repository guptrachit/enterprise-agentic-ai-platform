from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "openai"
    openai_api_key: str

    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_initial_backoff_seconds: float = 0.5
    llm_max_backoff_seconds: float = 5.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
