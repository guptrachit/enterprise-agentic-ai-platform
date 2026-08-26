from agent_platform.observability.operational_snapshot import (
    OperationalObservabilitySnapshot,
)


def test_operational_snapshot_to_dict() -> None:
    snapshot = OperationalObservabilitySnapshot(
        health={
            "status": "healthy",
            "reasons": [],
        },
        runtime={
            "resolved": True,
        },
        api={
            "total_requests": 10,
        },
        concurrency={
            "active_requests": 1,
        },
        rate_limit={
            "rejected_requests": 0,
        },
        security={
            "total_security_failures": 0,
        },
        export={
            "total_exports": 2,
            "successful_exports": 2,
            "failed_exports": 0,
        },
    )

    assert snapshot.to_dict() == {
        "health": {
            "status": "healthy",
            "reasons": [],
        },
        "runtime": {
            "resolved": True,
        },
        "api": {
            "total_requests": 10,
        },
        "concurrency": {
            "active_requests": 1,
        },
        "rate_limit": {
            "rejected_requests": 0,
        },
        "security": {
            "total_security_failures": 0,
        },
        "export": {
            "total_exports": 2,
            "successful_exports": 2,
            "failed_exports": 0,
        },
    }
