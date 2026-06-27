"""
Asset status endpoints — satellites, grid nodes, flight routes.

GET /api/status/satellites
GET /api/status/grid
GET /api/status/routes
GET /api/status/summary
"""

from flask import Blueprint, jsonify

from core.database import get_session
from core.models import Satellite, GridNode, FlightRoute, MitigationLog

bp = Blueprint("status", __name__)


@bp.get("/status/satellites")
def satellite_status():
    with get_session() as session:
        sats = session.query(Satellite).order_by(Satellite.id).all()
        return jsonify([
            {
                "id": s.id,
                "name": s.name,
                "orbit_type": s.orbit_type,
                "altitude_km": s.altitude_km,
                "status": s.status,
                "last_command": s.last_command,
                "last_command_at": s.last_command_at.isoformat() if s.last_command_at else None,
            }
            for s in sats
        ])


@bp.get("/status/grid")
def grid_status():
    with get_session() as session:
        nodes = session.query(GridNode).order_by(GridNode.gic_vulnerability.desc()).all()
        return jsonify([
            {
                "id": n.id,
                "name": n.name,
                "region": n.region,
                "node_type": n.node_type,
                "capacity_mw": n.capacity_mw,
                "gic_vulnerability": n.gic_vulnerability,
                "status": n.status,
                "last_action": n.last_action,
                "last_action_at": n.last_action_at.isoformat() if n.last_action_at else None,
            }
            for n in nodes
        ])


@bp.get("/status/routes")
def route_status():
    with get_session() as session:
        routes = session.query(FlightRoute).order_by(FlightRoute.id).all()
        return jsonify([
            {
                "id": r.id,
                "flight_number": r.flight_number,
                "origin": r.origin,
                "destination": r.destination,
                "route_type": r.route_type,
                "hf_dependency": r.hf_dependency,
                "status": r.status,
                "advisory": r.advisory,
                "advisory_issued_at": r.advisory_issued_at.isoformat() if r.advisory_issued_at else None,
            }
            for r in routes
        ])


@bp.get("/status/summary")
def status_summary():
    """High-level dashboard summary — asset counts by status."""
    with get_session() as session:
        # Satellites
        sat_counts: dict = {}
        for s in session.query(Satellite).all():
            sat_counts[s.status] = sat_counts.get(s.status, 0) + 1

        # Grid
        grid_counts: dict = {}
        for n in session.query(GridNode).all():
            grid_counts[n.status] = grid_counts.get(n.status, 0) + 1

        # Routes
        route_counts: dict = {}
        for r in session.query(FlightRoute).all():
            route_counts[r.status] = route_counts.get(r.status, 0) + 1

        # Latest brief
        latest_brief = (
            session.query(MitigationLog)
            .order_by(MitigationLog.created_at.desc())
            .first()
        )

        return jsonify({
            "satellites": sat_counts,
            "grid": grid_counts,
            "routes": route_counts,
            "latest_brief": {
                "severity": latest_brief.severity,
                "total_actions": latest_brief.total_actions_taken,
                "created_at": latest_brief.created_at.isoformat() if latest_brief.created_at else None,
                "executive_brief": latest_brief.executive_brief,
            } if latest_brief else None,
        })