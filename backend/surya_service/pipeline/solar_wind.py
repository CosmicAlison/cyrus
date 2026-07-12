"""
Solar Wind Speed Forecasting inference pipeline.

Task dir:  downstream_examples/solar_wind_forcasting/
Weights:   assets/solar_wind_weights.pth (nasa-ibm-ai4science/solar_wind_surya)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.base import BaseInferencePipeline

log = logging.getLogger(__name__)

CHECKPOINT = "assets/solar_wind_weights.pth"


class SolarWindPipeline(BaseInferencePipeline):
    task_dir    = "solar_wind_forcasting"
    result_type = "solar_wind"

    dataset_class_name = None   # UNVERIFIED — confirm via grep, see module docstring
    index_path_key      = None  # UNVERIFIED — falls back to generic {data_type}_data_path/valid_data_path chain
    rollout_steps        = 0    # UNVERIFIED — confirm against this task's own config/reference infer.py

    _wind_csv_df = None

    def _load_model(self) -> None:
        import torch
        from surya.utils.distributed import set_global_seed
        set_global_seed(42)

        self._config = self._load_config("config_infer.yaml")
        self._load_wind_csv()

        infer_module = self._import_task_module()
        checkpoint_path = str(self._task_path / CHECKPOINT)

        self._model = infer_module.load_model(
            config=self._config,
            checkpoint_path=checkpoint_path,
            device=torch.device(self.device),
        )
        self._model.eval()
        log.info("[solar_wind] model loaded from %s", checkpoint_path)

    def _resolve_wind_csv_path(self) -> str | None:
        data_cfg = self._config.get("data", {})
        # NOTE: base._load_config already resolves these to absolute paths
        # if they're present — this is just picking WHICH key holds the value.
        csv_path = (
            data_cfg.get("data_path")
            or data_cfg.get("csv_path")
            or data_cfg.get("wind_csv_path")
        )
        if csv_path:
            return csv_path

        candidates = sorted((self._task_path / "assets").rglob("*.csv"))
        return str(candidates[0]) if candidates else None

    def _load_wind_csv(self) -> None:
        import pandas as pd

        csv_path = self._resolve_wind_csv_path()
        if not csv_path:
            log.warning("[solar_wind] no wind CSV found - Bz/N will be 0")
            return

        try:
            self._wind_csv_df = pd.read_csv(csv_path, parse_dates=["Epoch"])
            self._wind_csv_df = self._wind_csv_df.sort_values("Epoch").reset_index(drop=True)
            log.info("[solar_wind] wind CSV loaded: %d rows", len(self._wind_csv_df))
        except Exception as exc:
            log.warning("[solar_wind] CSV load failed: %s", exc)

    def _load_timestamp_index(self) -> list[datetime]:
        csv_path = self._resolve_wind_csv_path()
        if not csv_path:
            return []
        return self._index_from_csv(csv_path, col="Epoch")

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
                    output = self._model(batch)  # (B, 1)

                speed_kms = float(output[0].to(torch.float32).cpu().item())

                ts_input  = _extract_ts(metadata, "timestamps_input",   0)
                ts_target = _extract_ts(metadata, "timestamps_targets",  0)

                wind_extras = self._lookup_wind_extras(ts_target)

                return {
                    "time_input":  ts_input,
                    "time_target": ts_target,
                    "speed_kms":   round(speed_kms, 2),
                    **wind_extras,
                }

        return _empty_wind_result()

    def _lookup_wind_extras(self, ts_target: str) -> dict[str, float]:
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
    import numpy as np
    try:
        arr = meta[key]
        val = arr[idx] if hasattr(arr, "__getitem__") else arr
        return str(np.datetime_as_string(np.asarray(val).ravel()[0], unit="m"))
    except Exception:
        return ""