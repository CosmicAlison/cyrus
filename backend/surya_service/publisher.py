"""
Publishes parsed forecast payloads to cyrus.raw_forecast queue.
"""

import json
import logging
from typing import Any

import pika

log = logging.getLogger(__name__)

RAW_FORECAST_QUEUE = "cyrus.raw_forecast"


def publish_raw_forecast(channel: pika.channel.Channel, payload: dict[str, Any]) -> None:
    channel.queue_declare(queue=RAW_FORECAST_QUEUE, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=RAW_FORECAST_QUEUE,
        body=json.dumps(payload).encode(),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent,
            content_type="application/json",
        ),
    )
    log.info("Published raw forecast for job %s", payload.get("job_id"))