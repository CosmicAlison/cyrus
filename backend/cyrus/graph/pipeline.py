"""
Compiled Cyrus LangGraph StateGraph.

Topology:
    START → helio_analyst
              ↓ (conditional fan-out)
    ┌─── satops ───┐
    ├─── gridops ──┼──→ commander → END
    └─── commsops ─┘

Ops agents that are not activated still pass through (they self-skip)
so the commander always receives three reports.

NOTE: LangGraph's Send API is used for true parallel fan-out.
Each ops node runs independently; commander waits for all three.
"""

import logging

from langgraph.graph import StateGraph, START, END

from cyrus.graph.state import CyrusState
from cyrus.graph.nodes import (
    helio_analyst_node,
    satops_node,
    gridops_node,
    commsops_node,
    commander_node,
)
from cyrus.graph.edges import route_after_analyst

log = logging.getLogger(__name__)


def build_pipeline():
    """Build and compile the Cyrus StateGraph."""
    builder = StateGraph(CyrusState)

    builder.add_node("helio_analyst", helio_analyst_node)
    builder.add_node("satops",        satops_node)
    builder.add_node("gridops",       gridops_node)
    builder.add_node("commsops",      commsops_node)
    builder.add_node("commander",     commander_node)

    builder.add_edge(START, "helio_analyst")

    builder.add_conditional_edges(
        "helio_analyst",
        route_after_analyst,
        {
            "satops":    "satops",
            "gridops":   "gridops",
            "commsops":  "commsops",
            "commander": "commander",  
        },
    )

    builder.add_edge("satops",   "commander")
    builder.add_edge("gridops",  "commander")
    builder.add_edge("commsops", "commander")

    builder.add_edge("commander", END)

    graph = builder.compile()
    log.info("Cyrus LangGraph pipeline compiled")
    return graph


cyrus_pipeline = build_pipeline()


def run_pipeline(raw_forecast: dict, threat_event_id: int) -> CyrusState:
    """
    Execute the full Cyrus pipeline for a given forecast.
    Returns the final state dict.
    """
    initial_state: CyrusState = {
        "raw_forecast": raw_forecast,
        "threat_event_id": threat_event_id,
    }
    log.info("Running Cyrus pipeline for job %s", raw_forecast.get("job_id"))
    final_state = cyrus_pipeline.invoke(initial_state)
    log.info("Pipeline complete for job %s", raw_forecast.get("job_id"))
    return final_state