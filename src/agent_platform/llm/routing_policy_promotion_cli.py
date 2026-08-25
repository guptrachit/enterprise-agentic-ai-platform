from agent_platform.llm.routing_policy_promotion import (
    RoutingPolicyPromotionDecision,
)


def promotion_exit_code(
    decision: RoutingPolicyPromotionDecision,
) -> int:
    """Translate a routing policy promotion decision into a process exit code."""

    return 0 if decision.allowed else 1
