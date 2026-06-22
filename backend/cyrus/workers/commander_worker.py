"""
workers/commander_worker.py
Consumes cyrus.agent_reports queue.
When all three ops agent reports for a job are received,
runs Agent 5 (Commander) to synthesise the executive brief.
"""

import logging
import time
from typing import Any

import pika

from cyrus.agents.commander import CommanderAgent
from cyrus.cache import redis_client
from cyrus.core.config import settings
from cyrus.core.database import get_session, init_db
from cyrus.core.models import MitigationLog, ForecastRun
from cyrus.core.schemas import AgentReport, ThreatPayload
from cyrus.messaging.consumer import BaseConsumer
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [commander_worker] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# How long to wait for all three reports before timing out (seconds)
REPORT_COLLECTION_TIMEOUT = 300
REPORT_POLL_INTERVAL = 3


class CommanderWorker(BaseConsumer):
    queue = settings.QUEUE_AGENT_REPORTS

    def __init__(self) -> None:
        super().__init__()
        self._agent = CommanderAgent()
        # Track which jobs we've already briefed to avoid duplicates
        self._briefed: set[str] = set()

    def handle_message(self, channel: pika.channel.Channel, payload: dict[str, Any]) -> None:
        job_id = payload.get("job_id", "unknown")
        agent = payload.get("agent", "unknown")

        if job_id in self._briefed:
            log.debug("[commander_worker] Job %s already briefed, ignoring %s report", job_id, agent)
            return

        log.info("[commander_worker] Received %s report for job %s", agent, job_id)

        # Store this report
        redis_client.store_agent_report(job_id, agent, payload)

        # Poll until all three reports are in or we time out
        if not self._wait_for_all_reports(job_id):
            log.warning("[commander_worker] Timed out waiting for all reports on job %s", job_id)

        # Synthesise even with partial reports (Commander handles missing ones gracefully)
        self._run_commander(job_id)

    def _wait_for_all_reports(self, job_id: str) -> bool:
        deadline = time.time() + REPORT_COLLECTION_TIMEOUT
        while time.time() < deadline:
            if redis_client.reports_complete(job_id):
                return True
            time.sleep(REPORT_POLL_INTERVAL)
        return False

    def _run_commander(self, job_id: str) -> None:
        if job_id in self._briefed:
            return
        self._briefed.add(job_id)

        reports_raw = redis_client.get_all_agent_reports(job_id)
        if not reports_raw:
            log.error("[commander_worker] No reports found for job %s", job_id)
            return

        redis_client.publish_dashboard_event("agent_started", {
            "job_id": job_id,
            "agent": "commander",
            "message": "Executive Commander synthesising mission report...",
        })

        # Reconstruct ThreatPayload from one of the reports (job_id is enough for lookup)
        threat_payload = self._load_threat_payload(job_id)
        if not threat_payload:
            log.error("[commander_worker] No ThreatPayload found for job %s", job_id)
            return

        reports = {
            agent: AgentReport.model_validate(report)
            for agent, report in reports_raw.items()
        }

        brief = self._agent.synthesise(threat_payload, reports)

        # Persist
        with get_session() as session:
            session.add(MitigationLog(
                forecast_run_id=job_id,
                executive_brief=brief.executive_brief,
                actions_summary={
                    "satops":   reports_raw.get("satops", {}).get("actions_taken", []),
                    "gridops":  reports_raw.get("gridops", {}).get("actions_taken", []),
                    "commsops": reports_raw.get("commsops", {}).get("actions_taken", []),
                },
                severity=brief.severity,
                total_actions_taken=brief.total_actions,
            ))
            run = session.get(ForecastRun, job_id)
            if run:
                run.status = "complete"
                run.completed_at = datetime.now(timezone.utc)

        redis_client.set_run_status(job_id, "complete")
        redis_client.publish_dashboard_event("pipeline_complete", {
            "job_id": job_id,
            "severity": brief.severity,
            "flare_class": brief.flare_class,
            "total_actions": brief.total_actions,
            "executive_brief": brief.executive_brief,
        })

        log.info(
            "[commander_worker] Brief complete for job %s — %d total actions",
            job_id, brief.total_actions,
        )

    def _load_threat_payload(self, job_id: str) -> ThreatPayload | None:
        """Load ThreatPayload from Postgres ThreatEvent."""
        try:
            from cyrus.core.models import ThreatEvent
            with get_session() as session:
                event = (
                    session.query(ThreatEvent)
                    .filter(ThreatEvent.forecast_run_id == job_id)
                    .order_by(ThreatEvent.id.desc())
                    .first()
                )
                if not event:
                    return None

                # The payload field stores the full ThreatPayload JSON
                # (written by helio_worker after analyst runs)
                raw = event.payload
                if "threat_payload" in raw:
                    return ThreatPayload.model_validate(raw["threat_payload"])
                # Fallback: reconstruct minimal ThreatPayload from event fields
                return ThreatPayload(
                    job_id=job_id,
                    threat_event_id=event.id,
                    severity=event.severity if event.severity != "pending" else "low",
                    flare_class=event.flare_class,
                    flare_probability=event.flare_probability,
                    euv_impact=event.euv_flux or 0.0,
                    magnetic_complexity=event.magnetic_complexity or 0.0,
                    solar_wind_speed_proxy=0.0,
                    atmospheric_drag_risk=event.euv_flux or 0.0,
                    active_region_x=0,
                    active_region_y=0,
                    peak_timestamp="",
                    analyst_summary="",
                    activate_satops=False,
                    activate_gridops=False,
                    activate_commsops=False,
                )
        except Exception as exc:
            log.error("[commander_worker] Failed to load ThreatPayload: %s", exc)
            return None


if __name__ == "__main__":
    init_db()
    CommanderWorker().run()