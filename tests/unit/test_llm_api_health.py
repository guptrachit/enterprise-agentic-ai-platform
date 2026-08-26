from unittest.mock import Mock

from agent_platform.llm.api_health import (
    create_llm_api_health_payload,
)


def test_llm_api_health_uses_observability_snapshot() -> None:
    observability_service = Mock()

    snapshot = Mock()

    snapshot.to_dict.return_value = {
        "health": {
            "status": "healthy",
            "reasons": [],
        },
        "runtime": {
            "resolved": True,
        },
        "api": {
            "total_requests": 0,
        },
        "concurrency": {
            "active_requests": 0,
        },
        "rate_limit": {
            "rejected_requests": 0,
        },
        "security": {
            "total_security_failures": 0,
        },
    }

    observability_service.snapshot.return_value = snapshot

    payload = create_llm_api_health_payload(
        observability_service=observability_service,
    )

    observability_service.snapshot.assert_called_once_with()

    assert payload == {
        "health": {
            "status": "healthy",
            "reasons": [],
        },
        "runtime": {
            "resolved": True,
        },
        "api": {
            "total_requests": 0,
        },
        "concurrency": {
            "active_requests": 0,
        },
        "rate_limit": {
            "rejected_requests": 0,
        },
        "security": {
            "total_security_failures": 0,
        },
    }
