"""
runner.py
---------
Surya service entry point.

Listens on RabbitMQ queue `cyrus.surya_jobs` for forecast job requests,
drives Surya inference via its CLI subprocess, parses the output NetCDF,
and publishes a structured forecast dict to `cyrus.raw_forecast`.

Job message schema (JSON):
    {
        "job_id":         str,
        "start_datetime": "YYYY-MM-DDTHH:MM:SS",   # UTC ISO format
        "end_datetime":   "YYYY-MM-DDTHH:MM:SS",
        "rollout_steps":  int                        # default 5
    }
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import pika

from model import run_inference
from parser import parse_prediction_nc
from publisher import publish_raw_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [surya_service] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
JOBS_QUEUE   = "cyrus.surya_jobs"
OUTPUT_BASE  = Path(os.environ.get("SURYA_OUTPUT_DIR", "/tmp/surya_output"))
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


# ── RabbitMQ connection ───────────────────────────────────────────────────────

def connect_rabbitmq(retries: int = 10, delay: int = 5) -> pika.BlockingConnection:
    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat = 600
    params.blocked_connection_timeout = 300

    for attempt in range(1, retries + 1):
        try:
            conn = pika.BlockingConnection(params)
            log.info("Connected to RabbitMQ")
            return conn
        except Exception as exc:
            log.warning("RabbitMQ not ready (attempt %d/%d): %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(delay)


# ── Message handler ───────────────────────────────────────────────────────────

def process_job(
    channel: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    _properties: pika.spec.BasicProperties,
    body: bytes,
) -> None:
    job = json.loads(body)
    job_id = job["job_id"]
    log.info("Received job %s", job_id)

    try:
        start_dt      = datetime.fromisoformat(job["start_datetime"])
        end_dt        = datetime.fromisoformat(job["end_datetime"])
        rollout_steps = int(job.get("rollout_steps", 5))

        job_output_dir = OUTPUT_BASE / job_id

        # ── Run Surya (subprocess) ────────────────────────────────────────────
        prediction_path = run_inference(
            start_dt=start_dt,
            end_dt=end_dt,
            rollout_steps=rollout_steps,
            output_dir=job_output_dir,
        )

        # ── Parse prediction.nc → structured dict ─────────────────────────────
        forecast_payload = parse_prediction_nc(
            nc_path=prediction_path,
            job_id=job_id,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        # ── Publish to cyrus.raw_forecast ─────────────────────────────────────
        publish_raw_forecast(channel, forecast_payload)

        channel.basic_ack(delivery_tag=method.delivery_tag)
        log.info("Job %s complete — forecast published", job_id)

    except Exception:
        log.error("Job %s failed:\n%s", job_id, traceback.format_exc())
        # Nack without requeue — goes to dead-letter for inspection
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    connection = connect_rabbitmq()
    channel    = connection.channel()

    channel.queue_declare(queue=JOBS_QUEUE, durable=True)
    # prefetch_count=1: Surya is GPU-bound, process one job at a time
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=JOBS_QUEUE,
        on_message_callback=process_job,
    )

    log.info("Surya service ready — waiting for jobs on %s", JOBS_QUEUE)
    channel.start_consuming()


if __name__ == "__main__":
    main()