"""
GET /api/stream — Server-Sent Events endpoint.

Subscribes to the Redis cyrus:dashboard pub/sub channel and
streams all pipeline events to connected clients in real time.

Event types published by workers:
  pipeline_started   — helio_worker kicked off the pipeline
  agent_started      — an individual agent has begun work
  agent_complete     — an individual agent has finished
  results_persisted  — DB write complete
  pipeline_complete  — final executive brief available
  pipeline_error     — something went wrong
"""

import json
import logging

from flask import Blueprint, Response, request, stream_with_context

from cache.redis_client import get_pubsub
from core.config import settings

log = logging.getLogger(__name__)

bp = Blueprint("stream", __name__)


@bp.get("/stream")
def stream():
    """
    SSE stream. Optionally filter by job_id via ?job_id=<id> query param.
    Clients receive all dashboard events by default.

    SSE format:
        data: {"type": "<event_type>", "data": {...}}\n\n
    """
    job_id_filter = request.args.get("job_id")

    def event_generator():
        pubsub = get_pubsub()
        pubsub.subscribe(settings.REDIS_DASHBOARD_CHANNEL)

        # Send a keepalive comment immediately so the client knows the connection is live
        yield ": connected\n\n"

        try:
            for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                raw = message["data"]
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Filter by job_id if requested
                if job_id_filter:
                    msg_job_id = parsed.get("data", {}).get("job_id")
                    if msg_job_id and msg_job_id != job_id_filter:
                        continue

                yield f"data: {raw}\n\n"

        except GeneratorExit:
            pubsub.unsubscribe(settings.REDIS_DASHBOARD_CHANNEL)
            pubsub.close()
        except Exception as exc:
            log.error("SSE stream error: %s", exc)
            pubsub.unsubscribe(settings.REDIS_DASHBOARD_CHANNEL)

    return Response(
        stream_with_context(event_generator()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": settings.FRONTEND_URL,
        },
    )