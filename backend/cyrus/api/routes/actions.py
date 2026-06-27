"""
GET /api/actions                        — list all agent actions (newest first)
GET /api/actions?job_id=<id>           — filter by forecast run
GET /api/actions?agent=<name>          — filter by agent
GET /api/actions/<forecast_run_id>     — all actions for a specific run
"""

from flask import Blueprint, jsonify, request

from core.database import get_session
from core.models import AgentAction

bp = Blueprint("actions", __name__)


@bp.get("/actions")
def list_actions():
    job_id = request.args.get("job_id")
    agent = request.args.get("agent")
    limit = min(int(request.args.get("limit", 100)), 500)

    with get_session() as session:
        q = session.query(AgentAction).order_by(AgentAction.created_at.desc())

        if job_id:
            q = q.filter(AgentAction.forecast_run_id == job_id)
        if agent:
            q = q.filter(AgentAction.agent == agent)

        actions = q.limit(limit).all()

        return jsonify([
            {
                "id": a.id,
                "forecast_run_id": a.forecast_run_id,
                "agent": a.agent,
                "action_type": a.action_type,
                "description": a.description,
                "details": a.details,
                "status": a.status,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in actions
        ])


@bp.get("/actions/<forecast_run_id>")
def get_run_actions(forecast_run_id: str):
    with get_session() as session:
        actions = (
            session.query(AgentAction)
            .filter(AgentAction.forecast_run_id == forecast_run_id)
            .order_by(AgentAction.created_at.asc())
            .all()
        )
        if not actions:
            return jsonify({"error": "No actions found for this run"}), 404

        return jsonify({
            "forecast_run_id": forecast_run_id,
            "total": len(actions),
            "actions": [
                {
                    "id": a.id,
                    "agent": a.agent,
                    "action_type": a.action_type,
                    "description": a.description,
                    "details": a.details,
                    "status": a.status,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in actions
            ],
        })