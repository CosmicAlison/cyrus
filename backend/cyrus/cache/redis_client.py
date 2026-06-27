import json
import logging
from typing import Any

import redis

from core.config import settings

log = logging.getLogger(__name__)

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _client


AGENT_REPORT_TTL = 3600  # 1 hour


def store_agent_report(job_id: str, agent: str, report: dict[str, Any]) -> None:
    """Store an agent's completed report so Commander can collect all three."""
    key = f"cyrus:reports:{job_id}:{agent}"
    get_client().set(key, json.dumps(report), ex=AGENT_REPORT_TTL)
    log.debug("Stored report: %s", key)


def get_agent_report(job_id: str, agent: str) -> dict[str, Any] | None:
    key = f"cyrus:reports:{job_id}:{agent}"
    raw = get_client().get(key)
    return json.loads(raw) if raw else None


def get_all_agent_reports(job_id: str) -> dict[str, dict[str, Any]]:
    """Returns reports for all three ops agents, or empty dict if not ready."""
    agents = ["satops", "gridops", "commsops"]
    results = {}
    for agent in agents:
        report = get_agent_report(job_id, agent)
        if report is not None:
            results[agent] = report
    return results


def reports_complete(job_id: str) -> bool:
    """True when all three ops agents have filed reports."""
    return len(get_all_agent_reports(job_id)) == 3




def set_run_status(job_id: str, status: str) -> None:
    get_client().set(f"cyrus:run:{job_id}:status", status, ex=AGENT_REPORT_TTL)


def get_run_status(job_id: str) -> str | None:
    return get_client().get(f"cyrus:run:{job_id}:status")




def publish_dashboard_event(event_type: str, data: dict[str, Any]) -> None:
    """Push an event to the dashboard SSE channel."""
    message = json.dumps({"type": event_type, "data": data})
    get_client().publish(settings.REDIS_DASHBOARD_CHANNEL, message)
    log.debug("Dashboard event: %s", event_type)


def get_pubsub() -> redis.client.PubSub:
    return get_client().pubsub()