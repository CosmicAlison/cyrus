"""
Drives Surya inference via its own CLI entry point:
    python easy_inference/run_easy_inference.py --config-path <yaml>

The Surya repo does NOT expose a stable Python import API — the model
architecture is loaded internally by run_easy_inference.py using its own
config-driven factory. We therefore drive it as a subprocess with a
generated YAML config and collect the output prediction.nc from the
configured output_dir.

Surya config key facts (from config_easy.yaml):
  - prompt_for_dates: false  → non-interactive, uses YAML dates
  - rollout_steps            → number of 12-min autoregressive steps
  - output_dir               → where prediction.nc is written
  - advanced.device: auto    → cuda → mps → cpu priority
  - advanced.weights_path    → auto-downloaded from HF on first run
  - cadence_minutes: 12      → SDO native cadence (override default of 60)
"""

import logging
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

SURYA_REPO = Path("/surya")
EASY_INFERENCE_SCRIPT = SURYA_REPO / "easy_inference" / "run_easy_inference.py"

# Model assets are auto-downloaded here by run_easy_inference.py
MODEL_DATA_DIR = SURYA_REPO / "data" / "Surya-1.0"


def build_config(
    start_dt: datetime,
    end_dt: datetime,
    rollout_steps: int,
    output_dir: Path,
    config_path: Path,
) -> Path:
    """
    Write a Surya config YAML for this job and return its path.
    Sets prompt_for_dates: false so the script runs non-interactively.
    """
    # Compute a validation_data_dir name that matches Surya's convention
    date_tag = start_dt.strftime("%Y%m%d_%Hmin")
    validation_data_dir = str(SURYA_REPO / f"data/Surya-1.0_validation_data_{date_tag}")

    config = {
        "user": {
            "start_datetime": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_datetime": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_for_dates": False,
            "output_dir": str(output_dir),
            "rollout_steps": rollout_steps,
        },
        "advanced": {
            "foundation_config_path": str(MODEL_DATA_DIR / "config.yaml"),
            "scalers_path": str(MODEL_DATA_DIR / "scalers.yaml"),
            "weights_path": str(MODEL_DATA_DIR / "surya.366m.v1.pt"),
            "model_repo_id": "nasa-ibm-ai4science/Surya-1.0",
            "model_allow_patterns": ["config.yaml", "scalers.yaml", "surya.366m.v1.pt"],
            "validation_data_dir": validation_data_dir,
            "index_path": str(output_dir / "index.csv"),
            "cadence_minutes": 12,
            "time_delta_input_minutes": [-12, 0],
            "time_delta_target_minutes": 12,
            "s3_bucket": "nasa-surya-bench",
            "download_skip_existing": True,
            "download_verify_size": False,
            "download_match_tolerance_minutes": 0,
            "prune_validation_data_to_window": False,
            "device": "auto",
            "dtype": "auto",
            "num_workers": 0,
            "prefetch_factor": 2,
            "gt_prefetch_workers": 4,
            "disable_autocast": False,
            "enable_tf32": True,
            "enable_cudnn_benchmark": True,
            "cpu_threads": 0,
            "show_progress": True,
            "debug_mode": False,
            "debug_log_path": str(output_dir / "inference_debug.txt"),
            "prediction_dtype": "float32",
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log.info("Surya config written to %s", config_path)
    return config_path


def run_inference(
    start_dt: datetime,
    end_dt: datetime,
    rollout_steps: int,
    output_dir: Path,
) -> Path:
    """
    Run Surya easy_inference and return the path to prediction.nc.

    Streams stdout/stderr from the subprocess to our logger so
    progress is visible in Docker logs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"

    build_config(
        start_dt=start_dt,
        end_dt=end_dt,
        rollout_steps=rollout_steps,
        output_dir=output_dir,
        config_path=config_path,
    )

    python_bin = SURYA_REPO / ".venv" / "bin" / "python"
    cmd = [
        str(python_bin),
        str(EASY_INFERENCE_SCRIPT),
        "--config-path",
        str(config_path),
    ]

    log.info("Running Surya: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=str(SURYA_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        log.info("[surya] %s", line.rstrip())

    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Surya inference failed with exit code {proc.returncode}. "
            f"Check logs above for details."
        )

    # run_easy_inference.py writes prediction.nc directly into output_dir
    prediction_path = output_dir / "prediction.nc"
    if not prediction_path.exists():
        # Some versions nest under a timestamped subfolder — find it
        candidates = list(output_dir.rglob("prediction.nc"))
        if not candidates:
            raise FileNotFoundError(
                f"prediction.nc not found under {output_dir} after Surya run"
            )
        prediction_path = candidates[0]
        log.info("prediction.nc found at: %s", prediction_path)

    return prediction_path