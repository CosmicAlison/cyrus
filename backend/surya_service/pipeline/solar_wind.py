"""
Solar Wind Speed Forecasting inference pipeline.

Task dir:  downstream_examples/solar_wind_forcasting/ 
Weights:   assets/solar_wind_weights.pth (nasa-ibm-ai4science/solar_wind_surya)
Coverage:  2010-05-01 to 2023-12-31 (best temporal coverage of all four tasks)

From docs:
  - Output: single continuous km/s value
  - Dataset CSV: Epoch, V, Bx, By, Bz, N columns
  - Input shape: (1, 13, 4096, 4096)
  - Prediction window: 4 days ahead (96 hours)
  - 5 physical quantities: V (km/s), Bx(GSE), By(GSM), Bz(GSM), N (density)

Output dict:
  {
    "time_input":    str,
    "time_target":   str,
    "speed_kms":     float,   # predicted solar wind speed km/s
    "bz_gsm":        float,   # from dataset label (measured, not predicted by image model)
    "bx_gse":        float,
    "by_gsm":        float,
    "density":       float,   # proton density N/cm³
  }

NOTE on Bz: The model predicts V (speed). Bx/By/Bz/N come from the CSV labels
for the target timestamp, they are ground-truth measurements at the target time,
not model predictions. This is still extremely useful: it tells us the actual
solar wind magnetic field conditions at the 4-day-ahead target, which directly
drives GIC risk in GridOps. 
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.base import BaseInferencePipeline, _utc

log = logging.getLogger(__name__)

CHECKPOINT = "assets/solar_wind_weights.pth"

# Typical solar wind speed range for normalisation reference
SPEED_MIN_KMS = 250.0
SPEED_MAX_KMS = 800.0


class SolarWindPipeline(BaseInferencePipeline):
    task_dir    = "solar_wind_forcasting"
    result_type = "solar_wind"

    # Cache the full CSV in memory for fast Bz/N lookup at target timestamp
    _wind_csv_df = None

    def _load_model(self) -> None:
        import torch
        from surya.utils.distributed import set_global_seed
        set_global_seed(42)

        self._config = self._load_config("config_infer.yaml")
        self._load_wind_csv()

        from infer import load_model
        checkpoint_path = str(self._task_path / CHECKPOINT)

        self._model = load_model(
            config=self._config,
            checkpoint_path=checkpoint_path,
            device=torch.device(self.device),
        )
        self._model.eval()
        log.info("[solar_wind] model loaded from %s", checkpoint_path)

    def _load_wind_csv(self) -> None:
        """Load the full wind CSV into memory for Bz/N lookup."""
        import pandas as pd

        data_cfg = self._config.get("data", {})
        csv_path = (
            data_cfg.get("data_path")
            or data_cfg.get("csv_path")
            or data_cfg.get("wind_csv_path")
        )

        if not csv_path:
            candidates = sorted((self._task_path / "assets").rglob("*.csv"))
            csv_path   = str(candidates[0]) if candidates else None

        if not csv_path:
            log.warning("[solar_wind] no wind CSV found — Bz/N will be 0")
            return

        full = Path(csv_path) if Path(csv_path).is_absolute() \
               else self._task_path / csv_path

        try:
            self._wind_csv_df = pd.read_csv(str(full), parse_dates=["Epoch"])
            self._wind_csv_df = self._wind_csv_df.sort_values("Epoch").reset_index(drop=True)
            log.info("[solar_wind] wind CSV loaded: %d rows", len(self._wind_csv_df))
        except Exception as exc:
            log.warning("[solar_wind] CSV load failed: %s", exc)

    def _load_timestamp_index(self) -> list[datetime]:
        """Load timestamps from the wind CSV Epoch column."""
        import pandas as pd

        data_cfg = self._config.get("data", {})
        csv_path = (
            data_cfg.get("data_path")
            or data_cfg.get("csv_path")
        )

        if not csv_path:
            candidates = sorted((self._task_path / "assets").rglob("*.csv"))
            csv_path   = str(candidates[0]) if candidates else None

        if not csv_path:
            return []

        full = Path(csv_path) if Path(csv_path).is_absolute() \
               else self._task_path / csv_path

        return self._index_from_csv(str(full), col="Epoch")

    def _run_at_index(self, sample_idx: int) -> dict[str, Any]:
        import torch

        loader, scalers = self._build_fixed_dataloader(sample_idx, data_type="test")

        with torch.no_grad():
            for batch, metadata in loader:
                if self.device != "cpu":
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.amp.autocast(
                    device_type="cuda" if "cuda" in self.device else "cpu",
                    dtype=self._config["dtype"],
                    enabled="cuda" in self.device,
                ):
                    output = self._model(batch)  # (B, 1) — single speed value

                speed_kms = float(output[0].cpu().item())

                ts_input  = _extract_ts(metadata, "timestamps_input",   0)
                ts_target = _extract_ts(metadata, "timestamps_targets",  0)

                # Look up Bz/N from CSV at the target timestamp
                wind_extras = self._lookup_wind_extras(ts_target)

                return {
                    "time_input":  ts_input,
                    "time_target": ts_target,
                    "speed_kms":   round(speed_kms, 2),
                    **wind_extras,
                }

        return _empty_wind_result()

    def _lookup_wind_extras(self, ts_target: str) -> dict[str, float]:
        """
        Look up Bz, Bx, By, N from the wind CSV at the closest timestamp
        to ts_target. These are measured values at the prediction target time.
        """
        if self._wind_csv_df is None or not ts_target:
            return {"bz_gsm": 0.0, "bx_gse": 0.0, "by_gsm": 0.0, "density": 0.0}

        try:
            import pandas as pd
            target = pd.Timestamp(ts_target, tz="UTC")
            df     = self._wind_csv_df
            idx    = (df["Epoch"] - target).abs().idxmin()
            row    = df.iloc[idx]
            return {
                "bz_gsm":  round(float(row.get("Bz", 0.0)), 3),
                "bx_gse":  round(float(row.get("Bx", 0.0)), 3),
                "by_gsm":  round(float(row.get("By", 0.0)), 3),
                "density": round(float(row.get("N",  0.0)), 3),
            }
        except Exception as exc:
            log.debug("[solar_wind] extras lookup failed: %s", exc)
            return {"bz_gsm": 0.0, "bx_gse": 0.0, "by_gsm": 0.0, "density": 0.0}


def _empty_wind_result() -> dict[str, Any]:
    return {
        "time_input": "", "time_target": "",
        "speed_kms": 0.0, "bz_gsm": 0.0,
        "bx_gse": 0.0, "by_gsm": 0.0, "density": 0.0,
    }


def _extract_ts(meta: dict, key: str, idx: int) -> str:
    try:
        arr = meta[key]
        val = arr[idx] if hasattr(arr, "__getitem__") else arr
        return str(np.datetime_as_string(np.asarray(val).ravel()[0], unit="m"))
    except Exception:
        return ""