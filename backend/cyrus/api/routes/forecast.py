import json
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from cache import redis_client
from core.config import settings
from core.database import get_session
from core.models import ForecastRun
from core.schemas import ForecastRequest, ForecastResponse
from messaging.broker import connect, declare_all_queues
from messaging.publisher import publish

bp = Blueprint("forecast", __name__)


@bp.post("/forecast")
def trigger_forecast():
    """
    Trigger a Surya inference job.

    Body (JSON):
        start_datetime: ISO 8601 UTC datetime (e.g. "2024-01-15T00:00:00")
        end_datetime:   ISO 8601 UTC datetime
        rollout_steps:  int, default 12 (number of 12-minute future steps to predict)

    Returns:
        job_id: str — use this to poll /api/forecast/<job_id> or subscribe to SSE
    """
    try:
        body = ForecastRequest.model_validate(request.get_json(force=True))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    if body.end_datetime <= body.start_datetime:
        return jsonify({"error": "end_datetime must be after start_datetime"}), 400

    job_id = str(uuid.uuid4())

    # Persist ForecastRun record
    with get_session() as session:
        session.add(ForecastRun(
            id=job_id,
            status="pending",
            forecast_start=body.start_datetime,
            forecast_end=body.end_datetime,
            rollout_steps=body.rollout_steps,
        ))

    redis_client.set_run_status(job_id, "pending")

    # Publish job to surya_service
    conn = connect()
    channel = conn.channel()
    declare_all_queues(channel)
    publish(channel, settings.QUEUE_SURYA_JOBS, {
        "job_id": job_id,
        "start_datetime": body.start_datetime.isoformat(),
        "end_datetime": body.end_datetime.isoformat(),
        "rollout_steps": body.rollout_steps,
    })
    conn.close()

    # Update run to surya_running
    with get_session() as session:
        run = session.get(ForecastRun, job_id)
        if run:
            run.status = "surya_running"

    redis_client.set_run_status(job_id, "surya_running")

    return jsonify(ForecastResponse(
        job_id=job_id,
        status="surya_running",
        message="Surya inference job queued. Subscribe to /api/stream for live updates.",
    ).model_dump()), 202


@bp.get("/forecast/<job_id>")
def get_forecast_status(job_id: str):
    """Poll the status of a forecast run."""
    # Fast path: Redis
    status = redis_client.get_run_status(job_id)
    if status:
        return jsonify({"job_id": job_id, "status": status})

    # Fallback: Postgres
    with get_session() as session:
        run = session.get(ForecastRun, job_id)
        if not run:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({
            "job_id": job_id,
            "status": run.status,
            "forecast_start": run.forecast_start.isoformat() if run.forecast_start else None,
            "forecast_end": run.forecast_end.isoformat() if run.forecast_end else None,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error": run.error,
        })


@bp.get("/forecast")
def list_forecasts():
    """List recent forecast runs (newest first, limit 20)."""
    with get_session() as session:
        runs = (
            session.query(ForecastRun)
            .order_by(ForecastRun.created_at.desc())
            .limit(20)
            .all()
        )
        return jsonify([
            {
                "job_id": r.id,
                "status": r.status,
                "forecast_start": r.forecast_start.isoformat() if r.forecast_start else None,
                "forecast_end": r.forecast_end.isoformat() if r.forecast_end else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in runs
        ])