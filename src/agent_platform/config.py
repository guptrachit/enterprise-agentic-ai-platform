from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_platform.llm.model_config import ModelConfig
from agent_platform.llm.model_policy_config import ModelPolicyConfig
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "openai"
    llm_model: str = "gpt-5-mini"
    llm_models: tuple[ModelConfig, ...] = ()
    llm_model_policy: ModelPolicyConfig = Field(default_factory=ModelPolicyConfig)
    openai_api_key: str

    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_initial_backoff_seconds: float = 0.5
    llm_max_backoff_seconds: float = 5.0

    llm_routing_policy_name: str = "production-routing-policy"

    llm_routing_policy_refresh_mode: RuntimePolicyRefreshMode = (
        RuntimePolicyRefreshMode.PROCESS_STATIC
    )

    llm_routing_policy_ttl_seconds: float | None = None

    llm_routing_refresh_metrics_enabled: bool = True

    @model_validator(mode="after")
    def validate_runtime_policy_refresh(
        self,
    ) -> "Settings":
        """Validate governed runtime policy refresh settings."""

        if not self.llm_routing_policy_name.strip():
            raise ValueError("llm_routing_policy_name must not be empty")

        if self.llm_routing_policy_refresh_mode is RuntimePolicyRefreshMode.TTL:
            if self.llm_routing_policy_ttl_seconds is None:
                raise ValueError(
                    "llm_routing_policy_ttl_seconds is required "
                    "when refresh mode is 'ttl'"
                )

            if self.llm_routing_policy_ttl_seconds <= 0:
                raise ValueError(
                    "llm_routing_policy_ttl_seconds must be greater than 0"
                )

        elif self.llm_routing_policy_ttl_seconds is not None:
            raise ValueError(
                "llm_routing_policy_ttl_seconds may only be "
                "configured when refresh mode is 'ttl'"
            )

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
