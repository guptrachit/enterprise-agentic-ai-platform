from dataclasses import dataclass
from enum import StrEnum


class RuntimePolicyRefreshMode(StrEnum):
    """Strategy for refreshing an active runtime routing policy."""

    PROCESS_STATIC = "process_static"
    PER_REQUEST = "per_request"
    TTL = "ttl"


@dataclass(frozen=True)
class RuntimePolicyRefreshConfig:
    """Configuration controlling runtime routing policy refresh."""

    mode: RuntimePolicyRefreshMode = RuntimePolicyRefreshMode.PROCESS_STATIC
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate refresh configuration."""

        if self.mode is RuntimePolicyRefreshMode.TTL:
            if self.ttl_seconds is None:
                raise ValueError(
                    "ttl_seconds is required when runtime policy refresh mode is 'ttl'."
                )

            if self.ttl_seconds <= 0:
                raise ValueError("ttl_seconds must be greater than 0.")

            return

        if self.ttl_seconds is not None:
            raise ValueError(
                "ttl_seconds may only be configured when "
                "runtime policy refresh mode is 'ttl'."
            )
