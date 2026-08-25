from enum import StrEnum


class RoutingPolicyLifecycleStatus(StrEnum):
    """Lifecycle status of a governed routing policy version."""

    DRAFT = "draft"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    ACTIVE = "active"
    RETIRED = "retired"
