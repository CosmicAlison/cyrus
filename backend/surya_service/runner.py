"""
Surya service entry point.

Listens on RabbitMQ queue `cyrus.surya_jobs` for forecast job requests,
runs Surya inference via the NASA-IMPACT model, parses the output NetCDF,
and publishes a structured forecast dict to `cyrus.raw_forecast`.

Job message schema (JSON):
    {
        "job_id": str,
        "start_datetime": "YYYY-MM-DDTHH:MM:SS",   # UTC
        "end_datetime":   "YYYY-MM-DDTHH:MM:SS",   # UTC
        "rollout_steps":  int                        # default 12
    }
"""

import json
import logging
import os
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import pika
import torch

from model import load_surya_model
from parser import parse_prediction_nc
from publisher import publish_raw_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [surya_service] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
JOBS_QUEUE = "cyrus.surya_jobs"
OUTPUT_DIR = Path(os.environ.get("SURYA_OUTPUT_DIR", "/tmp/surya_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", DEVICE)


def connect_rabbitmq(retries: int = 10, delay: int = 5) -> pika.BlockingConnection:
    params = pika.URLParameters(RABBITMQ_URL)
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


def process_job(model, channel, method, properties, body: bytes) -> None:
    job = json.loads(body)
    job_id = job["job_id"]
    log.info("Processing job %s", job_id)

    try:
        start_dt = datetime.fromisoformat(job["start_datetime"])
        end_dt = datetime.fromisoformat(job["end_datetime"])
        rollout_steps = job.get("rollout_steps", 12)

        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Run Surya inference
        prediction_path = run_surya_inference(
            model=model,
            start_dt=start_dt,
            end_dt=end_dt,
            rollout_steps=rollout_steps,
            output_dir=job_output_dir,
        )

        # Parse .nc output into structured dict
        forecast_payload = parse_prediction_nc(
            nc_path=prediction_path,
            job_id=job_id,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        # Publish to cyrus.raw_forecast
        publish_raw_forecast(channel, forecast_payload)

        channel.basic_ack(delivery_tag=method.delivery_tag)
        log.info("Job %s complete — forecast published", job_id)

    except Exception:
        log.error("Job %s failed:\n%s", job_id, traceback.format_exc())
        # Nack without requeue — dead-letter for inspection
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def run_surya_inference(
    model,
    start_dt: datetime,
    end_dt: datetime,
    rollout_steps: int,
    output_dir: Path,
) -> Path:
    """
    Run Surya forward pass and save prediction to NetCDF.

    The NASA-IMPACT Surya model accepts a sequence of multi-channel solar
    observation tensors and autoregressively predicts future states.

    Input channels (13 total):
        AIA: 94, 131, 171, 193, 211, 304, 335, 1600 Angstrom
        HMI: Bx, By, Bz (magnetogram), Dopplergram, Continuum

    The model is loaded via HuggingFace (nasa-ibm-ai4science/Surya-1.0).
    Data is fetched from the public SuryaBench S3 bucket.
    """
    from model import fetch_sdo_data, run_forward_pass, save_prediction_nc

    log.info("Fetching SDO data: %s → %s", start_dt.isoformat(), end_dt.isoformat())
    input_tensor = fetch_sdo_data(
        start_dt=start_dt,
        end_dt=end_dt,
        device=DEVICE,
    )

    log.info("Running Surya forward pass (%d rollout steps)", rollout_steps)
    prediction_tensor = run_forward_pass(
        model=model,
        input_tensor=input_tensor,
        rollout_steps=rollout_steps,
        device=DEVICE,
    )

    prediction_path = output_dir / "prediction.nc"
    save_prediction_nc(
        prediction=prediction_tensor,
        input_tensor=input_tensor,
        start_dt=start_dt,
        path=prediction_path,
    )

    log.info("Prediction saved: %s", prediction_path)
    return prediction_path


def main() -> None:
    log.info("Loading Surya model from HuggingFace...")
    model = load_surya_model(device=DEVICE)
    log.info("Model ready")

    connection = connect_rabbitmq()
    channel = connection.channel()

    channel.queue_declare(queue=JOBS_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)  # One job at a time (GPU bound)

    channel.basic_consume(
        queue=JOBS_QUEUE,
        on_message_callback=lambda ch, method, props, body: process_job(
            model, ch, method, props, body
        ),
    )

    log.info("Waiting for jobs on %s ...", JOBS_QUEUE)
    channel.start_consuming()


if __name__ == "__main__":
    main()