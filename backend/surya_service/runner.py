"""

Runs four Surya downstream model pipelines on a cadence timer,
aggregates results, and publishes to two RabbitMQ queues:

  cyrus.raw_forecast  — significant threat change → triggers full agent pipeline
  cyrus.telemetry     — every tick → lightweight dashboard update

Also listens on cyrus.surya_jobs for on-demand forecast requests
(e.g. triggered by POST /api/forecast from the Flask API).

Cadence:
  INFERENCE_INTERVAL_SECS  (default 600 = 10 min) — all four models
  Solar wind runs every 3rd tick (30 min) by default, otherwise cached.

Mode control:
  SURYA_USE_MOCK=true   — mock_generator.py, no GPU needed (local dev)
  SURYA_USE_MOCK=false  — real pipelines via AMD ROCm (MI300X)
"""

import json
import logging
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
import pika

from solar_clock import clock


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [surya_service] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

def _rocm_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
    
RABBITMQ_URL          = os.environ["RABBITMQ_URL"]
QUEUE_JOBS            = "cyrus.surya_jobs"
QUEUE_RAW_FORECAST    = "cyrus.raw_forecast"
QUEUE_TELEMETRY       = "cyrus.telemetry"
INFERENCE_INTERVAL    = int(os.environ.get("INFERENCE_INTERVAL_SECS", "600"))
WIND_EVERY_N_TICKS    = int(os.environ.get("WIND_EVERY_N_TICKS", "3"))
USE_MOCK              = os.environ.get("SURYA_USE_MOCK", "true").lower() == "true"
DEVICE                = os.environ.get("SURYA_DEVICE", "cuda" if _rocm_available() else "cpu")
log.info("Mode: %s | device: %s | cadence: %ds",
         "MOCK" if USE_MOCK else "REAL", DEVICE, INFERENCE_INTERVAL)





# Pipeline initialisation

def _init_pipelines():
    if USE_MOCK:
        return None

    log.info("Loading Surya downstream pipelines...")
    from pipeline.flare_forecast   import FlareForecastPipeline
    from pipeline.ar_segmentation  import ARSegmentationPipeline
    from pipeline.euv_spectra      import EUVSpectraPipeline
    from pipeline.solar_wind       import SolarWindPipeline

    pipelines = {
        "flare": FlareForecastPipeline(device=DEVICE),
        "ar":    ARSegmentationPipeline(device=DEVICE),
        "euv":   EUVSpectraPipeline(device=DEVICE),
        "wind":  SolarWindPipeline(device=DEVICE),
    }
    log.info("All pipelines ready")
    return pipelines


# Inference

def run_inference_tick(
    pipelines,
    tick: int,
    cached_wind: dict | None,
    job_id: str | None = None,
) -> tuple[dict, dict | None]:
    """
    Run one inference cycle. Returns (payload, cached_wind).
    pipelines is None in mock mode.
    """
    from aggregator import Aggregator

    solar_dt = clock.now()
    log.info("Tick %d | solar_time=%s", tick, solar_dt.isoformat())

    if USE_MOCK:
        from mock_generator import generate_all
        results = generate_all(solar_dt, tick, cached_wind)
    else:
        results = _run_real_pipelines(pipelines, solar_dt, tick, cached_wind)

    # Cache wind if we ran it this tick
    new_cached_wind = results["wind"] if not results["wind"].get("_cached") else cached_wind

    payload = _aggregator.build(
        solar_time=solar_dt,
        flare=results["flare"],
        ar_now=results["ar_now"],
        ar_flare=results["ar_flare"],
        euv=results["euv"],
        wind=results["wind"],
    )
    payload["job_id"] = job_id or str(uuid.uuid4())

    return payload, new_cached_wind


def _run_real_pipelines(
    pipelines: dict, solar_dt: datetime,
    tick: int, cached_wind: dict | None
) -> dict:
    """Run all four real Surya models. Solar wind cached every N ticks."""

    flare   = pipelines["flare"].infer(solar_dt)
    ar_now  = pipelines["ar"].infer(solar_dt)

    # AR call 2: only when flare is predicted
    ar_flare = None
    if flare.get("prediction") == 1 and flare.get("time_target"):
        try:
            target_dt = datetime.fromisoformat(
                flare["time_target"].replace("T", " ")
            ).replace(tzinfo=timezone.utc)
            ar_flare = pipelines["ar"].infer(target_dt)
            log.info("AR at flare time: %d regions", ar_flare.get("active_region_count", 0))
        except Exception as exc:
            log.warning("AR flare-time call failed: %s", exc)

    euv = pipelines["euv"].infer(solar_dt)

    # Wind: run every N ticks, otherwise use cache
    if tick % WIND_EVERY_N_TICKS == 0 or cached_wind is None:
        wind = pipelines["wind"].infer(solar_dt)
        wind["_cached"] = False
    else:
        wind = {**(cached_wind or {}), "_cached": True}
        log.debug("Wind cached (tick %d)", tick)

    return {
        "flare":    flare,
        "ar_now":   ar_now,
        "ar_flare": ar_flare,
        "euv":      euv,
        "wind":     wind,
    }


# RabbitMQ

def connect_rabbitmq(retries: int = 15, delay: int = 5) -> pika.BlockingConnection:
    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat = 600
    params.blocked_connection_timeout = 300

    for attempt in range(1, retries + 1):
        try:
            conn = pika.BlockingConnection(params)
            log.info("Connected to RabbitMQ at %s", RABBITMQ_URL)
            return conn
        except Exception as exc:
            log.warning("RabbitMQ attempt %d/%d: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(delay)


def publish(channel: pika.channel.Channel, queue: str, payload: dict) -> None:
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(payload, default=str).encode(),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent,
            content_type="application/json",
        ),
    )
    log.debug("Published to %s (job=%s)", queue, payload.get("job_id"))


def handle_job_request(
    channel, method, _props, body: bytes,
    pipelines, cached_wind_ref: list
) -> None:
    """Handle on-demand job from POST /api/forecast."""
    job = json.loads(body)
    job_id = job.get("job_id", str(uuid.uuid4()))
    log.info("On-demand job: %s", job_id)

    try:
        # Override clock if specific datetime requested
        requested = job.get("solar_datetime")
        if requested:
            solar_dt = datetime.fromisoformat(requested).replace(tzinfo=timezone.utc)
        else:
            solar_dt = clock.now()

        if USE_MOCK:
            from mock_generator import generate_all
            results = generate_all(solar_dt, 0, cached_wind_ref[0])
        else:
            results = _run_real_pipelines(pipelines, solar_dt, 0, cached_wind_ref[0])

        payload = _aggregator.build(
            solar_time=solar_dt,
            flare=results["flare"],
            ar_now=results["ar_now"],
            ar_flare=results["ar_flare"],
            euv=results["euv"],
            wind=results["wind"],
        )
        payload["job_id"] = job_id

        # On-demand always publishes to raw_forecast (triggers agents)
        publish(channel, QUEUE_RAW_FORECAST, payload)
        channel.basic_ack(delivery_tag=method.delivery_tag)
        log.info("On-demand job %s published", job_id)

    except Exception:
        log.error("On-demand job %s failed:\n%s", job_id, traceback.format_exc())
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)



_aggregator = None   # initialised in main()


def main() -> None:
    global _aggregator
    from aggregator import Aggregator
    _aggregator = Aggregator()

    pipelines   = _init_pipelines()
    connection  = connect_rabbitmq()
    channel     = connection.channel()

    for q in [QUEUE_JOBS, QUEUE_RAW_FORECAST, QUEUE_TELEMETRY]:
        channel.queue_declare(queue=q, durable=True)

    channel.basic_qos(prefetch_count=1)

    # Mutable reference so job handler can read/update cached wind
    cached_wind_ref = [None]

    # Register on-demand job consumer
    channel.basic_consume(
        queue=QUEUE_JOBS,
        on_message_callback=lambda ch, m, p, b: handle_job_request(
            ch, m, p, b, pipelines, cached_wind_ref
        ),
    )

    log.info("Surya service ready | cadence=%ds | mock=%s", INFERENCE_INTERVAL, USE_MOCK)
    log.info("Solar clock: %s", clock.info())

    tick = 0
    while True:
        tick_start = time.monotonic()

        # Process any queued on-demand jobs first (non-blocking)
        connection.process_data_events(time_limit=1)

        try:
            payload, new_wind = run_inference_tick(
                pipelines=pipelines,
                tick=tick,
                cached_wind=cached_wind_ref[0],
            )
            cached_wind_ref[0] = new_wind

            # Always publish telemetry (lightweight dashboard update)
            publish(channel, QUEUE_TELEMETRY, payload)

            # Only trigger agents if threat changed significantly
            if _aggregator.threat_delta_significant(payload):
                log.info("Threat delta significant — publishing to raw_forecast")
                publish(channel, QUEUE_RAW_FORECAST, payload)
            else:
                log.debug("No significant threat delta — telemetry only")

        except Exception:
            log.error("Tick %d failed:\n%s", tick, traceback.format_exc())

        tick += 1

        # Sleep for remainder of interval
        elapsed = time.monotonic() - tick_start
        sleep_for = max(0, INFERENCE_INTERVAL - elapsed)
        log.info("Tick %d done in %.1fs — sleeping %.0fs", tick - 1, elapsed, sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()