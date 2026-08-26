import pytest

from agent_platform.observability.exporter import (
    OperationalTelemetryExporter,
)


def test_operational_telemetry_exporter_is_abstract() -> None:
    with pytest.raises(
        TypeError,
    ):
        OperationalTelemetryExporter()
