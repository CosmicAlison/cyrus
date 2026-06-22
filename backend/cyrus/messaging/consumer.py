import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any

import pika

from cyrus.messaging.broker import connect, declare_all_queues

log = logging.getLogger(__name__)


class BaseConsumer(ABC):
    """
    Blocking RabbitMQ consumer with:
    - Automatic reconnection on connection drop
    - Structured JSON message parsing
    - Nack-without-requeue on unhandled exceptions
    - prefetch_count=1 (fair dispatch)
    """

    queue: str  # Subclasses set this

    def __init__(self) -> None:
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.channel.Channel | None = None

    def run(self) -> None:
        log.info("%s starting — consuming from %s", self.__class__.__name__, self.queue)
        while True:
            try:
                self._connection = connect()
                self._channel = self._connection.channel()
                declare_all_queues(self._channel)
                self._channel.basic_qos(prefetch_count=1)
                self._channel.basic_consume(
                    queue=self.queue,
                    on_message_callback=self._on_message,
                )
                self._channel.start_consuming()
            except pika.exceptions.AMQPConnectionError as exc:
                log.error("RabbitMQ connection lost: %s — reconnecting in 5s", exc)
                time.sleep(5)
            except KeyboardInterrupt:
                log.info("%s shutting down", self.__class__.__name__)
                break

    def _on_message(
        self,
        channel: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            payload = json.loads(body)
            self.handle_message(channel, payload)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            log.error(
                "%s failed to handle message:\n%s",
                self.__class__.__name__,
                traceback.format_exc(),
            )
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    @abstractmethod
    def handle_message(
        self,
        channel: pika.channel.Channel,
        payload: dict[str, Any],
    ) -> None:
        """Implement in subclass. Called once per message."""
        ...