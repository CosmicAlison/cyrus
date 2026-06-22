"""
LangGraph shared state schema for the Cyrus pipeline.

The StateGraph passes this dict between nodes. Each node reads
what it needs and writes its outputs back into the same dict.
"""

from typing import Any, TypedDict


class CyrusState(TypedDict, total=False):
    # Input — raw Surya forecast payload (from helio_worker)
    raw_forecast: dict[str, Any]

    # Set by helio_analyst node
    threat_payload: dict[str, Any]      # serialised ThreatPayload
    threat_event_id: int
    severity: str                        # low | moderate | high | extreme

    # Routing flags (set by helio_analyst, read by router edge)
    activate_satops: bool
    activate_gridops: bool
    activate_commsops: bool

    # Set by each ops agent node
    satops_report: dict[str, Any]       # serialised AgentReport
    gridops_report: dict[str, Any]
    commsops_report: dict[str, Any]

    # Set by commander node
    executive_brief: dict[str, Any]     # serialised ExecutiveBrief

    # Error tracking
    error: str | None