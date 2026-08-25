import json
import logging

from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.runtime_policy_refresh_telemetry import (
    create_runtime_policy_refresh_event,
    log_runtime_policy_refresh_event,
)


def test_refresh_event_preserves_values() -> None:
    event = create_runtime_policy_refresh_event(
        policy_name="production-policy",
        refresh_mode=RuntimePolicyRefreshMode.PER_REQUEST,
        policy_identifier="production-policy@1.1.0",
        refreshed=True,
        previous_policy_identifier="production-policy@1.0.0",
    )

    assert event.policy_name == "production-policy"
    assert event.refresh_mode == "per_request"

    assert event.policy_identifier == ("production-policy@1.1.0")

    assert event.refreshed is True

    assert event.previous_policy_identifier == ("production-policy@1.0.0")

    assert event.timestamp


def test_refresh_event_without_previous_policy() -> None:
    event = create_runtime_policy_refresh_event(
        policy_name="production-policy",
        refresh_mode=RuntimePolicyRefreshMode.PROCESS_STATIC,
        policy_identifier="production-policy@1.0.0",
        refreshed=True,
        previous_policy_identifier=None,
    )

    assert event.previous_policy_identifier is None


def test_refresh_event_logging(caplog) -> None:
    event = create_runtime_policy_refresh_event(
        policy_name="production-policy",
        refresh_mode=RuntimePolicyRefreshMode.TTL,
        policy_identifier="production-policy@2.0.0",
        refreshed=False,
        previous_policy_identifier="production-policy@2.0.0",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        log_runtime_policy_refresh_event(event)

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_runtime_policy_refresh ")
    ]

    assert len(records) == 1

    payload = json.loads(
        records[0].getMessage().removeprefix("llm_runtime_policy_refresh ")
    )

    assert payload["policy_identifier"] == ("production-policy@2.0.0")

    assert payload["refresh_mode"] == "ttl"
    assert payload["refreshed"] is False
