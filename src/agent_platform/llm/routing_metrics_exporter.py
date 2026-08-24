import json
import logging
from typing import Protocol

from agent_platform.llm.routing_metrics import RoutingMetricsSnapshot

logger = logging.getLogger("agent_platform.llm")


class RoutingMetricsExporter(Protocol):
    """Contract for exporting routing metrics snapshots."""

    def export(
        self,
        snapshot: RoutingMetricsSnapshot,
    ) -> None:
        """Export one routing metrics snapshot."""


class LoggingRoutingMetricsExporter:
    """Export routing metrics as structured JSON logs."""

    def export(
        self,
        snapshot: RoutingMetricsSnapshot,
    ) -> None:
        """Write a routing metrics snapshot to structured logs."""

        logger.info(
            "llm_routing_metrics %s",
            json.dumps(
                snapshot.to_dict(),
                sort_keys=True,
            ),
        )
