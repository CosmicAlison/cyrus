"""
GET /api/threats           — list all threat events
GET /api/threats/<id>      — single threat event detail
GET /api/threats/<id>/brief — executive brief for a threat's forecast run
"""

from flask import Blueprint, jsonify

from core.database import get_session
from core.models import ThreatEvent, MitigationLog

bp = Blueprint("threats", __name__)


@bp.get("/threats")
def list_threats():
    with get_session() as session:
        events = (
            session.query(ThreatEvent)
            .order_by(ThreatEvent.created_at.desc())
            .limit(50)
            .all()
        )
        return jsonify([
            {
                "id": e.id,
                "forecast_run_id": e.forecast_run_id,
                "severity": e.severity,
                "flare_class": e.flare_class,
                "flare_probability": e.flare_probability,
                "euv_flux": e.euv_flux,
                "magnetic_complexity": e.magnetic_complexity,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ])


@bp.get("/threats/<int:threat_id>")
def get_threat(threat_id: int):
    with get_session() as session:
        event = session.get(ThreatEvent, threat_id)
        if not event:
            return jsonify({"error": "Threat event not found"}), 404
        return jsonify({
            "id": event.id,
            "forecast_run_id": event.forecast_run_id,
            "severity": event.severity,
            "flare_class": event.flare_class,
            "flare_probability": event.flare_probability,
            "euv_flux": event.euv_flux,
            "magnetic_complexity": event.magnetic_complexity,
            "solar_wind_proxy": event.solar_wind_proxy,
            "payload": event.payload,
            "created_at": event.created_at.isoformat() if event.created_at else None,
        })


@bp.get("/threats/<int:threat_id>/brief")
def get_brief(threat_id: int):
    with get_session() as session:
        event = session.get(ThreatEvent, threat_id)
        if not event:
            return jsonify({"error": "Threat event not found"}), 404

        log = (
            session.query(MitigationLog)
            .filter(MitigationLog.forecast_run_id == event.forecast_run_id)
            .order_by(MitigationLog.created_at.desc())
            .first()
        )
        if not log:
            return jsonify({"error": "Executive brief not yet available"}), 404

        return jsonify({
            "forecast_run_id": event.forecast_run_id,
            "severity": log.severity,
            "executive_brief": log.executive_brief,
            "actions_summary": log.actions_summary,
            "total_actions_taken": log.total_actions_taken,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        })