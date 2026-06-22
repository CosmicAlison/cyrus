import json
import logging
from typing import Any

import pika

log = logging.getLogger(__name__)


def publish(
    channel: pika.channel.Channel,
    queue: str,
    payload: dict[str, Any],
) -> None:
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(payload, default=str).encode(),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent,
            content_type="application/json",
        ),
    )
    log.debug("Published to %s: job_id=%s", queue, payload.get("job_id", "?"))