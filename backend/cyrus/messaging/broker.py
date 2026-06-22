import logging
import time

import pika

from cyrus.core.config import settings

log = logging.getLogger(__name__)


def connect(
    retries: int = settings.RABBITMQ_CONNECT_RETRIES,
    delay: int = settings.RABBITMQ_CONNECT_DELAY,
) -> pika.BlockingConnection:
    params = pika.URLParameters(settings.RABBITMQ_URL)
    params.heartbeat = 600
    params.blocked_connection_timeout = 300

    for attempt in range(1, retries + 1):
        try:
            conn = pika.BlockingConnection(params)
            log.info("Connected to RabbitMQ")
            return conn
        except Exception as exc:
            log.warning(
                "RabbitMQ connection attempt %d/%d failed: %s",
                attempt, retries, exc,
            )
            if attempt == retries:
                raise
            time.sleep(delay)


def declare_all_queues(channel: pika.channel.Channel) -> None:
    """Idempotent queue declaration — safe to call on every startup."""
    queues = [
        settings.QUEUE_SURYA_JOBS,
        settings.QUEUE_RAW_FORECAST,
        settings.QUEUE_THREATS,
        settings.QUEUE_SATOPS,
        settings.QUEUE_GRIDOPS,
        settings.QUEUE_COMMSOPS,
        settings.QUEUE_AGENT_REPORTS,
    ]
    for q in queues:
        channel.queue_declare(queue=q, durable=True)
        log.debug("Queue declared: %s", q)