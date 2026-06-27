"""
LangChain tools for the SatOps agent (Agent 2).
Operates against the Satellite table in Postgres.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from cyrus.core.database import get_session
from cyrus.core.models import Satellite

log = logging.getLogger(__name__)


@tool
def query_satellites(orbit_type: str = "ALL") -> list[dict[str, Any]]:
    """
    Query the satellite registry. Returns satellite IDs, orbit type, altitude,
    and current status. Use orbit_type='LEO', 'MEO', 'GEO', or 'ALL'.
    """
    with get_session() as session:
        q = session.query(Satellite)
        if orbit_type != "ALL":
            q = q.filter(Satellite.orbit_type == orbit_type)
        return [
            {
                "id": s.id,
                "name": s.name,
                "orbit_type": s.orbit_type,
                "altitude_km": s.altitude_km,
                "inclination_deg": s.inclination_deg,
                "status": s.status,
            }
            for s in q.all()
        ]


@tool
def issue_safe_mode_command(satellite_id: str, reason: str) -> dict[str, Any]:
    """
    Command a satellite to enter Safe Mode: non-essential systems powered down,
    solar panels reoriented to minimise drag, no attitude manoeuvres.
    Use when flare probability > 0.45 or EUV spike is imminent.
    """
    with get_session() as session:
        sat = session.get(Satellite, satellite_id)
        if not sat:
            return {"success": False, "error": f"Satellite {satellite_id!r} not found"}

        sat.status = "safe_mode"
        sat.last_command = f"SAFE_MODE: {reason}"
        sat.last_command_at = datetime.now(timezone.utc)

    log.info("[satops_tool] %s → safe_mode | %s", satellite_id, reason)
    return {
        "success": True,
        "satellite_id": satellite_id,
        "new_status": "safe_mode",
        "command": "SAFE_MODE",
        "reason": reason,
    }


@tool
def adjust_orientation(satellite_id: str, target_attitude: str, reason: str) -> dict[str, Any]:
    """
    Command a satellite to adjust its orientation to minimise atmospheric drag
    or shield sensitive instruments. target_attitude examples: 'drag_minimise',
    'instrument_shield', 'nominal'.
    """
    with get_session() as session:
        sat = session.get(Satellite, satellite_id)
        if not sat:
            return {"success": False, "error": f"Satellite {satellite_id!r} not found"}

        sat.status = "reorienting"
        sat.last_command = f"ORIENT:{target_attitude} | {reason}"
        sat.last_command_at = datetime.now(timezone.utc)

    log.info("[satops_tool] %s → reorienting (%s)", satellite_id, target_attitude)
    return {
        "success": True,
        "satellite_id": satellite_id,
        "new_status": "reorienting",
        "target_attitude": target_attitude,
    }


@tool
def schedule_thruster_burn(
    satellite_id: str,
    delta_v_ms: float,
    burn_direction: str,
    reason: str,
) -> dict[str, Any]:
    """
    Schedule a thruster burn to raise or lower the satellite's orbit to avoid
    increased atmospheric drag from EUV heating.
    delta_v_ms: delta-v in m/s (positive = raise orbit, negative = lower).
    burn_direction: 'prograde' or 'retrograde'.
    Only use for LEO satellites where drag is a real concern.
    """
    with get_session() as session:
        sat = session.get(Satellite, satellite_id)
        if not sat:
            return {"success": False, "error": f"Satellite {satellite_id!r} not found"}
        if sat.orbit_type != "LEO":
            return {"success": False, "error": "Thruster burns are only scheduled for LEO satellites"}

        sat.status = "thruster_burn"
        sat.last_command = f"BURN:{burn_direction} dv={delta_v_ms}m/s | {reason}"
        sat.last_command_at = datetime.now(timezone.utc)

    log.info("[satops_tool] %s → thruster_burn %s %.2fm/s", satellite_id, burn_direction, delta_v_ms)
    return {
        "success": True,
        "satellite_id": satellite_id,
        "new_status": "thruster_burn",
        "delta_v_ms": delta_v_ms,
        "burn_direction": burn_direction,
    }


SATELLITE_TOOLS = [query_satellites, issue_safe_mode_command, adjust_orientation, schedule_thruster_burn]