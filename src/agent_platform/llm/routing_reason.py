from dataclasses import dataclass
from enum import StrEnum


class RoutingReasonCode(StrEnum):
    """Stable machine-readable codes describing model-routing decisions."""

    DISABLED = "disabled"
    WORKLOAD_NOT_SUPPORTED = "workload_not_supported"
    CONSTRAINT_REJECTED = "constraint_rejected"
    CAPABILITY_MISSING = "capability_missing"
    LOWER_COST_PREFERRED = "lower_cost_preferred"
    LOWER_LATENCY_PREFERRED = "lower_latency_preferred"
    PROVIDER_PREFERRED = "provider_preferred"
    COST_TIER_PREFERRED = "cost_tier_preferred"
    LATENCY_TIER_PREFERRED = "latency_tier_preferred"
    SELECTED = "selected"


@dataclass(frozen=True)
class RoutingReason:
    """Machine-readable routing reason with a human-readable explanation."""

    code: RoutingReasonCode
    message: str
