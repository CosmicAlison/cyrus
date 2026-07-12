"""
LangChain tools for the CommsOps agent (Agent 4).
Operates against the FlightRoute table in Postgres.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from core.database import get_session
from core.models import FlightRoute

log = logging.getLogger(__name__)


@tool
def query_flight_routes(route_type: str = "ALL") -> list[dict[str, Any]]:
    """
    Query active flight routes. Returns routes with their HF radio dependency
    level (high/medium/low) and current status.
    route_type: 'polar', 'oceanic', 'non-polar', or 'ALL'.
    Polar routes are most vulnerable to HF blackouts and radiation.
    """
    with get_session() as session:
        q = session.query(FlightRoute)
        if route_type != "ALL":
            q = q.filter(FlightRoute.route_type == route_type)
        return [
            {
                "id": r.id,
                "flight_number": r.flight_number,
                "origin": r.origin,
                "destination": r.destination,
                "route_type": r.route_type,
                "hf_dependency": r.hf_dependency,
                "status": r.status,
            }
            for r in q.all()
        ]


@tool
def get_high_risk_routes(hf_dependency: str = "high") -> list[dict[str, Any]]:
    """
    Get all flight routes with the specified HF radio dependency level.
    High-dependency polar routes are at risk of complete radio blackout
    during M/X-class flare events. These require immediate rerouting advisories.
    hf_dependency: 'high' or 'medium'.
    """
    with get_session() as session:
        routes = (
            session.query(FlightRoute)
            .filter(FlightRoute.hf_dependency == hf_dependency)
            .filter(FlightRoute.status == "nominal")
            .all()
        )
        return [
            {
                "id": r.id,
                "flight_number": r.flight_number,
                "origin": r.origin,
                "destination": r.destination,
                "route_type": r.route_type,
            }
            for r in routes
        ]


@tool
def issue_rerouting_advisory(
    route_id: str,
    advisory_text: str,
    alternate_route: str,
) -> dict[str, Any]:
    """
    Issue an HF blackout/radiation rerouting advisory for a flight route.

    Pushes a simulated NOTAM advisory to the affected route.
    advisory_text: explanation of the solar weather threat.
    alternate_route: recommended alternate flight path.
    """
    with get_session() as session:
        route = session.get(FlightRoute, route_id)
        if not route:
            return {"success": False, "error": f"Route {route_id!r} not found"}

        full_advisory = f"{advisory_text} ALTERNATE: {alternate_route}"
        route.status = "advisory_issued"
        route.advisory = full_advisory
        route.advisory_issued_at = datetime.now(timezone.utc)

        flight_number = route.flight_number

    log.info("[comms_tool] Advisory issued for route %s (%s)", route_id, flight_number)

    return {
        "success": True,
        "route_id": route_id,
        "flight_number": flight_number,
        "advisory": full_advisory,
        "status": "advisory_issued",
    }


@tool
def switch_to_backup_band(
    route_id: str,
    backup_system: str,
    reason: str,
) -> dict[str, Any]:
    """
    Switch a flight route's communications from HF radio to a backup system.
    backup_system options: 'SATCOM_L' (L-band satellite), 'SATCOM_KU' (Ku-band),
    'VHF_RELAY' (VHF ground relay where available).
    Use when HF blackout is expected during flare peak.
    """
    with get_session() as session:
        route = session.get(FlightRoute, route_id)
        if not route:
            return {"success": False, "error": f"Route {route_id!r} not found"}

        advisory = f"HF comms degraded: switched to {backup_system} | {reason}"
        route.status = "backup_comms"
        route.advisory = advisory
        route.advisory_issued_at = datetime.now(timezone.utc)

    log.info("[comms_tool] Route %s switched to %s", route_id, backup_system)
    return {
        "success": True,
        "route_id": route_id,
        "backup_system": backup_system,
        "new_status": "backup_comms",
    }


COMMS_TOOLS = [query_flight_routes, get_high_risk_routes, issue_rerouting_advisory, switch_to_backup_band]