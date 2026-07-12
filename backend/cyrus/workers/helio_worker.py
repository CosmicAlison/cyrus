"""
Consumes cyrus.raw_forecast from RabbitMQ.
Runs the full Cyrus LangGraph pipeline (all 5 agents).
Persists results to Postgres and publishes dashboard events to Redis.

This is the primary orchestration worker — it runs the complete
pipeline graph rather than just Agent 1 in isolation.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import pika

from cache import redis_client
from core.config import settings
from core.database import get_session, init_db
from core.models import ForecastRun, ThreatEvent, AgentAction, MitigationLog
from core.schemas import ThreatPayload, AgentReport, ExecutiveBrief
from graph.pipeline import run_pipeline
from messaging.consumer import BaseConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [helio_worker] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


class HelioWorker(BaseConsumer):
    queue = settings.QUEUE_RAW_FORECAST
    
    def handle_message(self, channel: pika.channel.Channel, payload: dict[str, Any]) -> None:
        job_id = payload.get("job_id", "unknown")

        _ensure_forecast_run(job_id, payload)
        log.info("Received raw forecast for job %s", job_id)

        # Update run status
        redis_client.set_run_status(job_id, "agents_running")
        redis_client.publish_dashboard_event("pipeline_started", {
            "job_id": job_id,
            "message": "Cyrus pipeline initiated — agents activating",
        })

        # Persist threat event (pre-agent-run, with raw data)
        threat_event_id = _persist_threat_event_placeholder(job_id, payload)

        try:
            # Run the full LangGraph pipeline
            final_state = run_pipeline(
                raw_forecast=payload,
                threat_event_id=threat_event_id,
            )

            brief = final_state.get("executive_brief")
            def bget(key, default=None):
                return brief.get(key, default) if isinstance(brief, dict) else getattr(brief, key, default)

            dominant = payload["ar_now"].get("dominant_region") or {"x": 0, "y": 0}
            lat, lon = to_lat_lon(dominant["x"], dominant["y"])

            active_regions = [
                {
                    "lat": to_lat_lon(c["x"], c["y"])[0],
                    "lon": to_lat_lon(c["x"], c["y"])[1],
                    "intensity": min(1.0, c.get("area_frac", 0) * 5),  # tune scale to taste
                }
                for c in payload["ar_now"].get("centroids", [])
            ]

            redis_client.set_run_status(job_id, "complete")
            redis_client.publish_dashboard_event("pipeline_complete", {
                "job_id": job_id,
                "severity": bget("severity", payload["threat_summary"]["severity"]),
                "flare_class": bget("flare_class", payload["flare"]["goes_class"]),
                "flare_probability": payload["flare"]["probability"],
                "total_actions": bget("total_actions", 0),
                "executive_brief": bget("executive_brief", ""),
                "active_region_lat": lat,
                "active_region_lon": lon,
                "active_regions": active_regions,
                "wind_speed": payload["wind"]["speed_kms"],
                "wind_density": payload["wind"]["density"],
            })
            log.info("Pipeline complete for job %s", job_id)

        except Exception as exc:
            log.error("Pipeline failed for job %s: %s", job_id, exc)
            redis_client.set_run_status(job_id, "failed")
            redis_client.publish_dashboard_event("pipeline_error", {
                "job_id": job_id,
                "error": str(exc),
            })
            _mark_run_failed(job_id, str(exc))
            raise

IMG_SIZE = 4096

def to_lat_lon(x: float, y: float) -> tuple[float, float]:
    """Rough disk-projected coords → approximate heliographic lat/lon.
    Accepts either normalized [-1,1] or raw pixel coords (auto-detected)."""
    if abs(x) > 1 or abs(y) > 1:
        # pixel space -> normalize first
        x = (x / IMG_SIZE) * 2 - 1
        y = (y / IMG_SIZE) * 2 - 1
    return round(y * 90, 1), round(x * 90, 1)

def _ensure_forecast_run( job_id: str, payload: dict) -> None:
        with get_session() as session:
            existing = session.get(ForecastRun, job_id)

            if existing:
                return

            now = datetime.now(timezone.utc)

            run = ForecastRun(
                id=job_id,
                status="surya_complete",
                forecast_start=now,
                forecast_end=now + timedelta(hours=2),
                rollout_steps=12,
            )

            session.add(run)
            session.commit()

            log.info(
                "[helio_worker] Created synthetic ForecastRun %s",
                job_id,
            )

def _persist_threat_event_placeholder(job_id: str, raw: dict) -> int:
    """Create a ThreatEvent row before agents run so we have an ID to reference."""
    summary = raw.get("summary", {})
    with get_session() as session:
        event = ThreatEvent(
            forecast_run_id=job_id,
            severity="pending",
            flare_class=summary.get("peak_flare_class", "A"),
            flare_probability=summary.get("max_flare_probability", 0.0),
            euv_flux=summary.get("mean_euv_flux"),
            magnetic_complexity=summary.get("max_magnetic_complexity"),
            payload=raw,
        )
        session.add(event)
        session.flush()
        return event.id


def _persist_results(job_id: str, threat_event_id: int, state: dict) -> None:
    """Write all agent actions and the executive brief to Postgres."""
    with get_session() as session:

        # Update ThreatEvent with assessed severity
        event = session.get(ThreatEvent, threat_event_id)
        if event and state.get("threat_payload"):
            tp = state["threat_payload"]
            event.severity = tp.get("severity", "low")
            event.flare_class = tp.get("flare_class", "A")
            event.flare_probability = tp.get("flare_probability", 0.0)

        # Persist per-agent actions
        for agent in ["satops", "gridops", "commsops"]:
            report_key = f"{agent}_report"
            report = state.get(report_key)
            if not report:
                continue
            for action in report.get("actions_taken", []):
                session.add(AgentAction(
                    forecast_run_id=job_id,
                    threat_event_id=threat_event_id,
                    agent=agent,
                    action_type=action.get("tool", "unknown"),
                    description=str(action.get("result", "")),
                    details=action,
                    status="success",
                ))

        # Persist executive brief
        brief = state.get("executive_brief")
        if brief:
            session.add(MitigationLog(
                forecast_run_id=job_id,
                executive_brief=brief.get("executive_brief", ""),
                actions_summary={
                    "satops":   state.get("satops_report", {}).get("actions_taken", []),
                    "gridops":  state.get("gridops_report", {}).get("actions_taken", []),
                    "commsops": state.get("commsops_report", {}).get("actions_taken", []),
                },
                severity=brief.get("severity", "low"),
                total_actions_taken=brief.get("total_actions", 0),
            ))

        # Mark ForecastRun complete
        run = session.get(ForecastRun, job_id)
        if run:
            run.status = "complete"
            run.completed_at = datetime.now(timezone.utc)

        # Publish each agent action to dashboard as it's saved
        redis_client.publish_dashboard_event("results_persisted", {
            "job_id": job_id,
            "threat_event_id": threat_event_id,
        })


def _mark_run_failed(job_id: str, error: str) -> None:
    try:
        with get_session() as session:
            run = session.get(ForecastRun, job_id)
            if run:
                run.status = "failed"
                run.error = error
                run.completed_at = datetime.now(timezone.utc)
    except Exception as e:
        log.error("Failed to mark run as failed: %s", e)


if __name__ == "__main__":
    init_db()
    HelioWorker().run()