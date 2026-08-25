from agent_platform.llm.routing_deployment_gate import (
    RoutingDeploymentGateResult,
)


def deployment_gate_exit_code(
    result: RoutingDeploymentGateResult,
) -> int:
    """Translate a routing deployment gate result into a process exit code."""

    return 0 if result.passed else 1
