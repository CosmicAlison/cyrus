"""
LangChain tools for the GridOps agent (Agent 3).
Operates against the GridNode table in Postgres.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from core.database import get_session
from core.models import GridNode

log = logging.getLogger(__name__)


@tool
def query_grid_topology(region: str = "ALL") -> list[dict[str, Any]]:
    """
    Query the power grid topology. Returns nodes with their GIC vulnerability
    scores (0-1, higher = more exposed to geomagnetically induced currents).
    Use region='ALL' or a specific region name.
    """
    with get_session() as session:
        q = session.query(GridNode)
        if region != "ALL":
            q = q.filter(GridNode.region == region)
        nodes = q.order_by(GridNode.gic_vulnerability.desc()).all()
        return [
            {
                "id": n.id,
                "name": n.name,
                "region": n.region,
                "node_type": n.node_type,
                "capacity_mw": n.capacity_mw,
                "gic_vulnerability": n.gic_vulnerability,
                "status": n.status,
            }
            for n in nodes
        ]


@tool
def get_high_risk_nodes(vulnerability_threshold: float = 0.7) -> list[dict[str, Any]]:
    """
    Get all grid nodes above a GIC vulnerability threshold.
    Default threshold 0.7 identifies nodes at serious risk of transformer damage.
    These should be prioritised for protective action during CME events.
    """
    with get_session() as session:
        nodes = (
            session.query(GridNode)
            .filter(GridNode.gic_vulnerability >= vulnerability_threshold)
            .filter(GridNode.status == "online")
            .order_by(GridNode.gic_vulnerability.desc())
            .all()
        )
        return [
            {
                "id": n.id,
                "name": n.name,
                "region": n.region,
                "node_type": n.node_type,
                "capacity_mw": n.capacity_mw,
                "gic_vulnerability": n.gic_vulnerability,
            }
            for n in nodes
        ]


@tool
def reroute_load(
    node_id: str,
    load_reduction_percent: float,
    reason: str,
) -> dict[str, Any]:
    """
    Reduce load on a grid node by a given percentage to protect against GIC surge.
    Triggers dynamic load redistribution to adjacent, lower-vulnerability nodes.
    load_reduction_percent: 0-100, how much to reduce load on this node.
    Use for nodes with vulnerability 0.5-0.75 where decoupling is premature.
    """
    with get_session() as session:
        node = session.get(GridNode, node_id)
        if not node:
            return {"success": False, "error": f"Node {node_id!r} not found"}

        node.status = "load_reduced"
        node.last_action = f"LOAD_REDUCE:{load_reduction_percent:.0f}% | {reason}"
        node.last_action_at = datetime.now(timezone.utc)

    reduced_mw = node.capacity_mw * (load_reduction_percent / 100)
    log.info("[grid_tool] %s → load_reduced by %.0f%%", node_id, load_reduction_percent)
    return {
        "success": True,
        "node_id": node_id,
        "new_status": "load_reduced",
        "load_reduction_percent": load_reduction_percent,
        "estimated_mw_rerouted": reduced_mw,
    }


@tool
def decouple_transformer(node_id: str, reason: str) -> dict[str, Any]:
    """
    Disconnect a high-vulnerability transformer from the grid to prevent
    GIC-induced thermal runaway and permanent damage.
    Use ONLY for nodes with vulnerability > 0.8 when a severe CME is imminent.
    This is a significant protective action — the node will be offline until manually restored.
    """
    with get_session() as session:
        node = session.get(GridNode, node_id)
        if not node:
            return {"success": False, "error": f"Node {node_id!r} not found"}
        if node.node_type != "transformer":
            return {"success": False, "error": f"Decoupling is only valid for transformers, not {node.node_type!r}"}

        node.status = "decoupled"
        node.last_action = f"DECOUPLE | {reason}"
        node.last_action_at = datetime.now(timezone.utc)

    log.warning("[grid_tool] %s DECOUPLED from grid | %s", node_id, reason)
    return {
        "success": True,
        "node_id": node_id,
        "new_status": "decoupled",
        "reason": reason,
        "warning": "Manual restoration required after event",
    }


GRID_TOOLS = [query_grid_topology, get_high_risk_nodes, reroute_load, decouple_transformer]