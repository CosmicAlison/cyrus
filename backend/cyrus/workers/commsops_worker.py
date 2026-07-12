"""
Standalone CommsOps worker — consumes cyrus.commsops queue.
"""

import logging
from typing import Any

import pika

from agents.commsops import CommsOpsAgent
from cache import redis_client
from core.config import settings
from core.database import get_session, init_db
from core.models import AgentAction
from core.schemas import ThreatPayload
from messaging.consumer import BaseConsumer
from messaging.publisher import publish

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [commsops_worker] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


class CommsOpsWorker(BaseConsumer):
    queue = settings.QUEUE_COMMSOPS

    def __init__(self) -> None:
        super().__init__()
        self._agent = CommsOpsAgent()

    def handle_message(self, channel: pika.channel.Channel, payload: dict[str, Any]) -> None:
        job_id = payload.get("job_id", "unknown")
        log.info("[commsops_worker] Processing threat for job %s", job_id)

        redis_client.publish_dashboard_event("agent_started", {
            "job_id": job_id,
            "agent": "commsops",
            "message": "CommsOps specialist assessing HF blackout and radiation risk...",
        })

        threat = ThreatPayload.model_validate(payload)
        report = self._agent.respond(threat)

        with get_session() as session:
            for action in report.actions_taken:
                session.add(AgentAction(
                    forecast_run_id=job_id,
                    agent="commsops",
                    action_type=action.get("tool", "unknown"),
                    description=str(action.get("result", "")),
                    details=action,
                    status=report.status,
                ))

        redis_client.store_agent_report(job_id, "commsops", report.model_dump())
        publish(channel, settings.QUEUE_AGENT_REPORTS, report.model_dump())

        redis_client.publish_dashboard_event("agent_complete", {
            "job_id": job_id,
            "agent": "commsops",
            "status": report.status,
            "message": report.summary,
            "actions_count": len(report.actions_taken),
        })

        log.info(
            "[commsops_worker] Job %s complete — %d actions, status=%s",
            job_id, len(report.actions_taken), report.status,
        )


if __name__ == "__main__":
    init_db()
    CommsOpsWorker().run()