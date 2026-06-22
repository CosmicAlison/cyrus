"""
Standalone SatOps worker — consumes cyrus.satops queue.

This worker runs Agent 2 in isolation (without the full pipeline).
Used when you want to scale SatOps independently or replay ThreatPayloads.
The helio_worker runs the full pipeline; this is the standalone fallback.
"""

import logging
from typing import Any

import pika

from cyrus.agents.satops import SatOpsAgent
from cyrus.cache import redis_client
from cyrus.core.config import settings
from cyrus.core.database import get_session, init_db
from cyrus.core.models import AgentAction
from cyrus.core.schemas import ThreatPayload
from cyrus.messaging.consumer import BaseConsumer
from cyrus.messaging.publisher import publish

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [satops_worker] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


class SatOpsWorker(BaseConsumer):
    queue = settings.QUEUE_SATOPS

    def __init__(self) -> None:
        super().__init__()
        self._agent = SatOpsAgent()

    def handle_message(self, channel: pika.channel.Channel, payload: dict[str, Any]) -> None:
        job_id = payload.get("job_id", "unknown")
        log.info("[satops_worker] Processing threat for job %s", job_id)

        redis_client.publish_dashboard_event("agent_started", {
            "job_id": job_id,
            "agent": "satops",
            "message": "SatOps engineer assessing satellite risk...",
        })

        threat = ThreatPayload.model_validate(payload)
        report = self._agent.respond(threat)

        # Persist actions
        with get_session() as session:
            for action in report.actions_taken:
                session.add(AgentAction(
                    forecast_run_id=job_id,
                    agent="satops",
                    action_type=action.get("tool", "unknown"),
                    description=action.get("result", ""),
                    details=action,
                    status=report.status,
                ))

        # Store report in Redis for Commander collection
        redis_client.store_agent_report(job_id, "satops", report.model_dump())

        # Publish to agent_reports queue
        publish(channel, settings.QUEUE_AGENT_REPORTS, report.model_dump())

        redis_client.publish_dashboard_event("agent_complete", {
            "job_id": job_id,
            "agent": "satops",
            "status": report.status,
            "message": report.summary,
            "actions_count": len(report.actions_taken),
        })

        log.info(
            "[satops_worker] Job %s complete — %d actions, status=%s",
            job_id, len(report.actions_taken), report.status,
        )


if __name__ == "__main__":
    init_db()
    SatOpsWorker().run()