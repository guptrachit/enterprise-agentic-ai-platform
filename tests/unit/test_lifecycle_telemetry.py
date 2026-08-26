import json
import logging

from agent_platform.lifecycle_telemetry import (
    LifecycleEventType,
    create_lifecycle_event,
    log_lifecycle_event,
)


def test_lifecycle_event_preserves_values() -> None:
    event = create_lifecycle_event(
        event_type=LifecycleEventType.STARTUP_SUCCEEDED,
        app_env="production",
    )

    assert event.event_type is LifecycleEventType.STARTUP_SUCCEEDED

    assert event.app_env == "production"
    assert event.timestamp


def test_lifecycle_event_is_logged(
    caplog,
) -> None:
    event = create_lifecycle_event(
        event_type=LifecycleEventType.SHUTDOWN_COMPLETED,
        app_env="development",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.lifecycle",
    ):
        log_lifecycle_event(event)

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("lifecycle_event ")
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("lifecycle_event "))

    assert payload["event_type"] == ("application_shutdown_completed")

    assert payload["app_env"] == "development"


def test_lifecycle_event_does_not_expose_secrets(
    caplog,
) -> None:
    event = create_lifecycle_event(
        event_type=LifecycleEventType.STARTUP_SUCCEEDED,
        app_env="production",
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.lifecycle",
    ):
        log_lifecycle_event(event)

    serialized = "\n".join(record.getMessage() for record in caplog.records).lower()

    assert "openai_api_key" not in serialized
    assert "authorization:" not in serialized
    assert "bearer " not in serialized
    assert "prompt" not in serialized
